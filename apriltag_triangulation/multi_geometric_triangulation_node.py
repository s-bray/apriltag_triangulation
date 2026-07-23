#!/usr/bin/env python3
"""
multi_geometric_triangulation_node.py — Method B generalized to N cameras.

Least-squares ray intersection: the published position is the point p that
minimizes the summed squared perpendicular distance to all viewing rays:

    minimize  Σ_i ‖(I − d_i d_iᵀ)(p − C_i)‖²
    →  A p = b   with  A = Σ (I − d_i d_iᵀ),  b = Σ (I − d_i d_iᵀ) C_i

With 2 rays this reduces to the classic common-perpendicular midpoint;
with 3+ it is overdetermined and the RMS point-to-ray residual becomes a
stronger consistency metric than the 2-ray gap.

Scale still comes from the camera baselines (positions in TF), never from
fx·tag_size — the same scale-independence as the 2-camera version.
baseline_override generalizes to scaling ALL camera positions about cam1
by (override / current cam1↔cam2 distance), preserving the geometry shape
while anchoring its size to a tape measurement.

Publishes:
  /apriltag/geometric_pose  (PoseStamped)  LSQ position; orientation from cam1
  /apriltag/ray_gap         (Float32)      RMS point-to-ray residual (m)
  /apriltag/scale_check     (Float32)      ‖geometric‖ / ‖fused‖
"""

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
import numpy as np
import cv2
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32
from apriltag_msgs.msg import AprilTagDetectionArray


class MultiGeometricTriangulationNode(Node):

    def __init__(self):
        super().__init__('multi_geometric_triangulation_node')

        self.declare_parameter('n_cams', 3)
        self.declare_parameter('tag_id', 0)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('max_age_sec', 1.0)
        self.declare_parameter('min_decision_margin', 50.0)
        self.declare_parameter('min_rays', 2)
        self.declare_parameter('baseline_override', 0.0)  # tape cam1<->cam2 (m)

        self.n           = self.get_parameter('n_cams').value
        self.tag_id      = self.get_parameter('tag_id').value
        self.world_frame = self.get_parameter('world_frame').value
        rate_hz          = self.get_parameter('rate_hz').value
        self.max_age     = self.get_parameter('max_age_sec').value
        self.min_margin  = self.get_parameter('min_decision_margin').value
        self.min_rays    = max(2, self.get_parameter('min_rays').value)
        self.baseline_override = self.get_parameter('baseline_override').value

        self.frames = {i: f'cam{i}_optical_frame'
                       for i in range(1, self.n + 1)}

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.K = {i: None for i in range(1, self.n + 1)}
        self.D = {i: None for i in range(1, self.n + 1)}
        self.det = {i: None for i in range(1, self.n + 1)}
        self.fused_latest = None
        self.cam1_pose_latest = None

        for i in range(1, self.n + 1):
            self.create_subscription(
                CameraInfo, f'/cam{i}/camera_info',
                lambda m, idx=i: self._info_cb(idx, m), 10)
            self.create_subscription(
                AprilTagDetectionArray, f'/cam{i}/apriltag/detections',
                lambda m, idx=i: self._det_cb(idx, m), 10)
        self.create_subscription(PoseStamped, '/apriltag/triangulated_pose',
                                 self._fused_cb, 10)
        self.create_subscription(PoseStamped, '/cam1/apriltag_pose',
                                 self._cam1_cb, 10)

        self.pub_pose  = self.create_publisher(
            PoseStamped, '/apriltag/geometric_pose', 10)
        self.pub_dist  = self.create_publisher(
            Float32, '/apriltag/geometric_distance', 10)
        self.pub_gap   = self.create_publisher(Float32, '/apriltag/ray_gap', 10)
        self.pub_scale = self.create_publisher(Float32, '/apriltag/scale_check', 10)

        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f'Multi-camera geometric triangulation ready ({self.n} cameras, '
            f'min_rays={self.min_rays})')

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _info_cb(self, i, msg):
        if self.K[i] is None:
            self.K[i] = np.array(msg.k).reshape(3, 3)
            self.D[i] = np.array(msg.d)

    def _det_cb(self, i, msg): self.det[i] = msg
    def _fused_cb(self, msg):  self.fused_latest = msg
    def _cam1_cb(self, msg):   self.cam1_pose_latest = msg

    # ── Helpers ───────────────────────────────────────────────────────────
    def _fresh(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return (now - t) < self.max_age

    def _centre(self, i):
        msg = self.det[i]
        if msg is None or not self._fresh(msg):
            return None
        for d in msg.detections:
            if d.id == self.tag_id and d.decision_margin >= self.min_margin:
                return (d.centre.x, d.centre.y)
        return None

    def _extrinsic(self, i):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.frames[i], rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        C = np.array([t.x, t.y, t.z])
        x, y, z, w = q.x, q.y, q.z, q.w
        Rm = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
        ])
        return C, Rm

    def _ray(self, i):
        if self.K[i] is None:
            return None
        px = self._centre(i)
        if px is None:
            return None
        ext = self._extrinsic(i)
        if ext is None:
            return None
        C, Rm = ext
        und = cv2.undistortPoints(
            np.array([[[px[0], px[1]]]], dtype=np.float64),
            self.K[i], self.D[i])
        d = np.array([float(und[0, 0, 0]), float(und[0, 0, 1]), 1.0])
        d /= np.linalg.norm(d)
        return C, Rm @ d

    # ── Main tick ─────────────────────────────────────────────────────────
    def _tick(self):
        rays = {}
        for i in range(1, self.n + 1):
            r = self._ray(i)
            if r is not None:
                rays[i] = r
        if len(rays) < self.min_rays:
            return

        # Optional: anchor scale to the tape-measured cam1<->cam2 distance
        if self.baseline_override > 0 and 1 in rays and 2 in rays:
            C1 = rays[1][0]
            cur = np.linalg.norm(rays[2][0] - C1)
            if cur > 1e-9:
                s = self.baseline_override / cur
                rays = {i: (C1 + (C - C1) * s, d)
                        for i, (C, d) in rays.items()}

        # Least-squares intersection:  A p = b
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for C, d in rays.values():
            P = np.eye(3) - np.outer(d, d)     # projector ⟂ to the ray
            A += P
            b += P @ C
        try:
            p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.get_logger().warn('Degenerate ray geometry',
                                   throttle_duration_sec=5.0)
            return

        # RMS point-to-ray residual
        res = []
        for C, d in rays.values():
            v = p - C
            res.append(np.linalg.norm(v - np.dot(v, d) * d))
        rms = float(np.sqrt(np.mean(np.square(res))))

        g = Float32(); g.data = rms
        self.pub_gap.publish(g)

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.world_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, p)
        if self.cam1_pose_latest is not None \
                and self._fresh(self.cam1_pose_latest):
            ps.pose.orientation = self.cam1_pose_latest.pose.orientation
        else:
            ps.pose.orientation.w = 1.0
        self.pub_pose.publish(ps)

        # Distance from world origin (= cam1 lens): √(x²+y²+z²).
        dist = Float32()
        dist.data = float(np.linalg.norm(p))
        self.pub_dist.publish(dist)

        if self.fused_latest is not None and self._fresh(self.fused_latest):
            f = self.fused_latest.pose.position
            nf = float(np.linalg.norm([f.x, f.y, f.z]))
            if nf > 1e-6:
                sc = Float32()
                sc.data = float(np.linalg.norm(p)) / nf
                self.pub_scale.publish(sc)


def main(args=None):
    rclpy.init(args=args)
    node = MultiGeometricTriangulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
