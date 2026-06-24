#!/usr/bin/env python3
"""
apriltag_adapter_node.py

apriltag_ros publishes detections directly into /tf (not a topic).
This adapter node:
  1. Listens to the TF tree for tag_<id> frames appearing under the camera frame
  2. Converts them to geometry_msgs/PoseStamped on a topic
  3. Also listens to /apriltag/detections (AprilTagDetectionArray) for
     the decision_margin so we can flag low-confidence detections

Publishes:
  /camX/apriltag_pose  (geometry_msgs/PoseStamped) — tag pose in camera frame
  /camX/apriltag_detected (std_msgs/Bool)          — True when tag is visible

One instance runs per camera namespace.
"""

import rclpy
from rclpy.node import Node
import rclpy.duration
import tf2_ros
import tf2_geometry_msgs  # noqa: F401
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from apriltag_msgs.msg import AprilTagDetectionArray


class AprilTagAdapterNode(Node):

    def __init__(self):
        super().__init__('apriltag_adapter_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('tag_id',       0)
        self.declare_parameter('tag_family',   '36h11')
        self.declare_parameter('camera_frame', 'cam1_optical_frame')
        self.declare_parameter('detections_topic', '/apriltag/detections')
        self.declare_parameter('min_decision_margin', 50.0)  # confidence threshold
        self.declare_parameter('publish_rate_hz', 20.0)

        self.tag_id        = self.get_parameter('tag_id').value
        self.tag_family    = self.get_parameter('tag_family').value
        self.camera_frame  = self.get_parameter('camera_frame').value
        det_topic          = self.get_parameter('detections_topic').value
        self.min_margin    = self.get_parameter('min_decision_margin').value
        rate_hz            = self.get_parameter('publish_rate_hz').value

        # TF child frame published by apriltag_ros
        # self.tag_frame = f'tag{self.tag_family}:{self.tag_id}'
        self.tag_frame = f'tag{self.tag_id}'


        # ── TF ───────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Detection confidence tracking ─────────────────────────────────────
        self.tag_visible = False
        self.create_subscription(
            AprilTagDetectionArray, det_topic,
            self._detection_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_pose     = self.create_publisher(PoseStamped, 'apriltag_pose',     10)
        self.pub_detected = self.create_publisher(Bool,        'apriltag_detected', 10)

        # ── Timer ─────────────────────────────────────────────────────────────
        self.create_timer(1.0 / rate_hz, self._publish)

        self.get_logger().info(
            f'AprilTag adapter: watching {self.tag_frame} '
            f'in frame {self.camera_frame}')

    def _detection_cb(self, msg: AprilTagDetectionArray):
        """Track whether tag is currently detected with sufficient confidence."""
        self.tag_visible = False
        for det in msg.detections:
            if det.id == self.tag_id:
                if det.decision_margin >= self.min_margin:
                    self.tag_visible = True
                break

    def _publish(self):
        # Publish detected flag
        b = Bool()
        b.data = self.tag_visible
        self.pub_detected.publish(b)

        if not self.tag_visible:
            return

        # Look up tag pose in camera frame from TF
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.tag_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return

        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = self.camera_frame
        ps.pose.position.x = tf_stamped.transform.translation.x
        ps.pose.position.y = tf_stamped.transform.translation.y
        ps.pose.position.z = tf_stamped.transform.translation.z
        ps.pose.orientation = tf_stamped.transform.rotation

        self.pub_pose.publish(ps)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
