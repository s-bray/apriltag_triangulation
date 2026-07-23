#!/usr/bin/env python3
"""
apriltag_triangulation_node.py

Dual-camera triangulation for AprilTag ground truth tracking.
No Kalman filter (raw fused pose only) — add later if needed.

Subscribes to:
  /cam1/apriltag_pose  (geometry_msgs/PoseStamped) — from apriltag_adapter_node
  /cam2/apriltag_pose  (geometry_msgs/PoseStamped)

Publishes:
  /apriltag/triangulated_pose       (geometry_msgs/PoseStamped) — fused
  /apriltag/cam1_pose               (geometry_msgs/PoseStamped)
  /apriltag/cam2_pose               (geometry_msgs/PoseStamped)
  /apriltag/triangulation_error     (std_msgs/Float32) — metres
  /apriltag/triangulation_error_pct (std_msgs/Float32) — % of baseline
  /apriltag/camera_baseline         (std_msgs/Float32) — metres (once)
"""

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
import numpy as np
import tf2_ros
import tf2_geometry_msgs  # noqa: F401
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation as R


# ─────────────────────────────────────────────────────────────────────────────
# SE(3) helpers
# ─────────────────────────────────────────────────────────────────────────────

def pose_to_matrix(pose):
    t = np.array([pose.position.x, pose.position.y, pose.position.z])
    q = [pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


def matrix_to_pose(T):
    from geometry_msgs.msg import Pose
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(T[0,3]), float(T[1,3]), float(T[2,3])
    q = R.from_matrix(T[:3, :3]).as_quat()
    p.orientation.x, p.orientation.y = float(q[0]), float(q[1])
    p.orientation.z, p.orientation.w = float(q[2]), float(q[3])
    return p


def average_poses(T1, T2, w1=0.5, w2=0.5):
    t_avg = w1 * T1[:3, 3] + w2 * T2[:3, 3]
    q1 = R.from_matrix(T1[:3, :3]).as_quat()
    q2 = R.from_matrix(T2[:3, :3]).as_quat()
    if np.dot(q1, q2) < 0:
        q2 = -q2
    q_avg = w1 * q1 + w2 * q2
    q_avg /= np.linalg.norm(q_avg)
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q_avg).as_matrix()
    T[:3, 3] = t_avg
    return T


def pose_distance(T1, T2):
    return float(np.linalg.norm(T1[:3, 3] - T2[:3, 3]))


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class AprilTagTriangulationNode(Node):

    def __init__(self):
        super().__init__('apriltag_triangulation_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('world_frame',      'world')
        self.declare_parameter('cam1_frame',       'cam1_optical_frame')
        self.declare_parameter('cam2_frame',       'cam2_optical_frame')
        self.declare_parameter('max_discrepancy_m', 0.15)
        self.declare_parameter('cam1_topic',       '/cam1/apriltag_pose')
        self.declare_parameter('cam2_topic',       '/cam2/apriltag_pose')
        self.declare_parameter('fusion_rate_hz',   20.0)

        self.world_frame     = self.get_parameter('world_frame').value
        self.cam1_frame      = self.get_parameter('cam1_frame').value
        self.cam2_frame      = self.get_parameter('cam2_frame').value
        self.max_discrepancy = self.get_parameter('max_discrepancy_m').value
        cam1_topic           = self.get_parameter('cam1_topic').value
        cam2_topic           = self.get_parameter('cam2_topic').value
        rate_hz              = self.get_parameter('fusion_rate_hz').value

        # ── TF ───────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── State ─────────────────────────────────────────────────────────────
        self.latest_cam1: PoseStamped | None = None
        self.latest_cam2: PoseStamped | None = None
        self.max_age_sec = 0.5
        self.baseline_m: float | None = None

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(PoseStamped, cam1_topic, self._cb_cam1, 10)
        self.create_subscription(PoseStamped, cam2_topic, self._cb_cam2, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_fused     = self.create_publisher(PoseStamped, '/apriltag/triangulated_pose',       10)
        self.pub_dist      = self.create_publisher(Float32,     '/apriltag/triangulated_distance',   10)
        self.pub_cam1      = self.create_publisher(PoseStamped, '/apriltag/cam1_pose',               10)
        self.pub_cam2      = self.create_publisher(PoseStamped, '/apriltag/cam2_pose',               10)
        self.pub_error     = self.create_publisher(Float32,     '/apriltag/triangulation_error',     10)
        self.pub_error_pct = self.create_publisher(Float32,     '/apriltag/triangulation_error_pct', 10)
        self.pub_baseline  = self.create_publisher(Float32,     '/apriltag/camera_baseline',         10)

        self.create_timer(1.0 / rate_hz, self._fuse)

        self.get_logger().info('AprilTag triangulation node ready')
        self.get_logger().info(f'  cam1: {cam1_topic}')
        self.get_logger().info(f'  cam2: {cam2_topic}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_cam1(self, msg): self.latest_cam1 = msg
    def _cb_cam2(self, msg): self.latest_cam2 = msg

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_fresh(self, msg: PoseStamped) -> bool:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        msg_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return (now_sec - msg_sec) < self.max_age_sec

    def _transform_to_world(self, ps: PoseStamped):
        """Transform PoseStamped from its frame to world frame. Returns 4x4 or None."""
        try:
            ps_w = self.tf_buffer.transform(
                ps, self.world_frame,
                timeout=rclpy.duration.Duration(seconds=0.1))
            return pose_to_matrix(ps_w.pose)
        except Exception as e:
            self.get_logger().warn(f'TF transform failed: {e}')
            return None

    def _stamped(self, T: np.ndarray) -> PoseStamped:
        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = self.world_frame
        ps.pose = matrix_to_pose(T)
        return ps

    def _get_baseline(self) -> float | None:
        if self.baseline_m is not None:
            return self.baseline_m
        try:
            t1 = self.tf_buffer.lookup_transform(
                self.world_frame, self.cam1_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
            t2 = self.tf_buffer.lookup_transform(
                self.world_frame, self.cam2_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
            p1 = np.array([t1.transform.translation.x,
                           t1.transform.translation.y,
                           t1.transform.translation.z])
            p2 = np.array([t2.transform.translation.x,
                           t2.transform.translation.y,
                           t2.transform.translation.z])
            self.baseline_m = float(np.linalg.norm(p1 - p2))
            self.get_logger().info(
                f'Camera baseline: {self.baseline_m:.4f} m')
            msg = Float32()
            msg.data = self.baseline_m
            self.pub_baseline.publish(msg)
            return self.baseline_m
        except Exception as e:
            self.get_logger().warn(f'Baseline lookup failed: {e}')
            return None

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _fuse(self):
        # Collect fresh poses
        ps1 = self.latest_cam1 if (self.latest_cam1 and self._is_fresh(self.latest_cam1)) else None
        ps2 = self.latest_cam2 if (self.latest_cam2 and self._is_fresh(self.latest_cam2)) else None

        if ps1 is None and ps2 is None:
            return

        # Transform to world
        T1 = self._transform_to_world(ps1) if ps1 else None
        T2 = self._transform_to_world(ps2) if ps2 else None

        # Publish individual
        if T1 is not None:
            self.pub_cam1.publish(self._stamped(T1))
        if T2 is not None:
            self.pub_cam2.publish(self._stamped(T2))

        # Fuse
        if T1 is not None and T2 is not None:
            discrepancy = pose_distance(T1, T2)

            e = Float32(); e.data = discrepancy
            self.pub_error.publish(e)

            baseline = self._get_baseline()
            if baseline and baseline > 0.0:
                pct = Float32()
                pct.data = float((discrepancy / baseline) * 100.0)
                self.pub_error_pct.publish(pct)

            # if discrepancy > self.max_discrepancy:
            #     self.get_logger().warn(
            #         f'Discrepancy {discrepancy:.3f} m > {self.max_discrepancy} m '
            #         f'— using cam1 only')
            #     T_fused = T1
            # else:
            #     T_fused = average_poses(T1, T2)
            T_fused = average_poses(T1, T2) #* 0.8 # MAKE SURE YOU ADJUST THIS RATIO

        elif T1 is not None:
            T_fused = T1
        else:
            T_fused = T2

        self.pub_fused.publish(self._stamped(T_fused))

        # Distance from world origin (= cam1 lens): √(x²+y²+z²).
        # Directly comparable to a laser/tape measurement cam1 → tag centre.
        dist = Float32()
        dist.data = float(np.linalg.norm(T_fused[:3, 3]))
        self.pub_dist.publish(dist)

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagTriangulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
