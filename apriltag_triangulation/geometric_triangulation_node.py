#!/usr/bin/env python3
"""
geometric_triangulation_node.py

TRUE geometric triangulation for the dual-camera AprilTag rig — a scale-
independent cross-check on the pose-fusion pipeline.

Method:
  1. Take the tag's detected CENTRE PIXEL from each camera's
     AprilTagDetectionArray (not the PnP pose!).
  2. Undistort it and back-project a 3D viewing ray from each camera,
     using only intrinsics ANGLES (K, D) — no fx·L depth involved.
  3. Express both rays in the world frame via the static camera TFs.
  4. Find the midpoint of the common perpendicular between the two rays
     → 3D position. Depth scale comes from the CAMERA BASELINE, not from
     tag_size × focal length.

Why this matters: the pose-fusion pipeline's scale rides on fx·L, so any
intrinsics/tag_size scale error shifts it. This node's scale rides on the
baseline instead. Comparing the two outputs therefore measures the fx·L
scale error live:  /apriltag/geometric_pose  vs  /apriltag/triangulated_pose.

baseline_override (optional, metres): if > 0, cam2's position is rescaled
along the cam1→cam2 direction so the baseline equals this tape-measured
value. For RAY intersection this is legitimate (unlike for pose fusion):
no per-camera depths are consumed, so anchoring the geometry to a physical
tape measurement introduces no inconsistency — it makes the output's scale
traceable to your tape instead of to the PnP-derived extrinsic length.

Publishes:
  /apriltag/geometric_pose  (PoseStamped) — ray-intersection position;
                             orientation copied from cam1's PnP pose
  /apriltag/ray_gap         (Float32) — distance between the two rays at
                             closest approach (m). Live quality metric:
                             grows with extrinsic-rotation error and
                             pixel noise. Healthy: < ~1-2 cm.
  /apriltag/scale_check     (Float32) — |geometric| / |fused| distance
                             ratio when both available; directly reads
                             the fx·L scale error (1.00 = no error).
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


class GeometricTriangulationNode(Node):

    def __init__(self):
        super().__init__('geometric_triangulation_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('tag_id', 0)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('cam1_frame', 'cam1_optical_frame')
        self.declare_parameter('cam2_frame', 'cam2_optical_frame')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('max_age_sec', 0.2)
        self.declare_parameter('min_decision_margin', 50.0)
        # 0.0 = use TF baseline as-is; >0 = rescale cam2 position so the
        # baseline equals this tape-measured value (see module docstring).
        self.declare_parameter('baseline_override', 0.0)

        self.tag_id       = self.get_parameter('tag_id').value
        self.world_frame  = self.get_parameter('world_frame').value
        self.cam1_frame   = self.get_parameter('cam1_frame').value
        self.cam2_frame   = self.get_parameter('cam2_frame').value
        rate_hz           = self.get_parameter('rate_hz').value
        self.max_age_sec  = self.get_parameter('max_age_sec').value
        self.min_margin   = self.get_parameter('min_decision_margin').value
        self.baseline_override = self.get_parameter('baseline_override').value

        # ── TF ────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Per-camera state ──────────────────────────────────────────────
        self.K = {1: None, 2: None}          # 3x3 intrinsics
        self.D = {1: None, 2: None}          # distortion coeffs
        self.det = {1: None, 2: None}        # latest detection msg
        self.fused_latest: PoseStamped | None = None
        self.cam1_pose_latest: PoseStamped | None = None  # for orientation

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(CameraInfo, '/cam1/camera_info',
                                 lambda m: self._info_cb(1, m), 10)
        self.create_subscription(CameraInfo, '/cam2/camera_info',
                                 lambda m: self._info_cb(2, m), 10)
        self.create_subscription(AprilTagDetectionArray,
                                 '/cam1/apriltag/detections',
                                 lambda m: self._det_cb(1, m), 10)
        self.create_subscription(AprilTagDetectionArray,
                                 '/cam2/apriltag/detections',
                                 lambda m: self._det_cb(2, m), 10)
        self.create_subscription(PoseStamped, '/apriltag/triangulated_pose',
                                 self._fused_cb, 10)
        self.create_subscription(PoseStamped, '/cam1/apriltag_pose',
                                 self._cam1_pose_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_pose  = self.create_publisher(
            PoseStamped, '/apriltag/geometric_pose', 10)
        self.pub_dist  = self.create_publisher(
            Float32, '/apriltag/geometric_distance', 10)
        self.pub_gap   = self.create_publisher(
            Float32, '/apriltag/ray_gap', 10)
        self.pub_scale = self.create_publisher(
            Float32, '/apriltag/scale_check', 10)

        self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info('Geometric triangulation node ready')
        if self.baseline_override > 0:
            self.get_logger().info(
                f'  baseline anchored to tape value: '
                f'{self.baseline_override:.4f} m')

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _info_cb(self, cam, msg: CameraInfo):
        if self.K[cam] is None:
            self.K[cam] = np.array(msg.k).reshape(3, 3)
            self.D[cam] = np.array(msg.d)

    def _det_cb(self, cam, msg): self.det[cam] = msg
    def _fused_cb(self, msg):    self.fused_latest = msg
    def _cam1_pose_cb(self, msg): self.cam1_pose_latest = msg

    # ── Helpers ───────────────────────────────────────────────────────────
    def _fresh(self, msg) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return (now - t) < self.max_age_sec

    def _centre_pixel(self, cam):
        """Return (u, v) of the tag centre if freshly detected, else None."""
        msg = self.det[cam]
        if msg is None or not self._fresh(msg):
            return None
        for d in msg.detections:
            if d.id == self.tag_id and d.decision_margin >= self.min_margin:
                return (d.centre.x, d.centre.y)
        return None

    def _cam_extrinsic(self, frame):
        """Return (C, Rm): camera position and rotation in world, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        C = np.array([t.x, t.y, t.z])
        # quaternion (x,y,z,w) → rotation matrix
        x, y, z, w = q.x, q.y, q.z, q.w
        Rm = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
        ])
        return C, Rm

    def _ray(self, cam, frame):
        """World-frame viewing ray (origin C, unit direction d) or None."""
        if self.K[cam] is None:
            return None
        px = self._centre_pixel(cam)
        if px is None:
            return None
        ext = self._cam_extrinsic(frame)
        if ext is None:
            return None
        C, Rm = ext

        # Undistort the centre pixel → normalized image coords (x_n, y_n)
        pts = np.array([[[px[0], px[1]]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, self.K[cam], self.D[cam])
        x_n, y_n = float(und[0, 0, 0]), float(und[0, 0, 1])

        d_cam = np.array([x_n, y_n, 1.0])
        d_cam /= np.linalg.norm(d_cam)
        d_world = Rm @ d_cam
        return C, d_world

    # ── Main tick ─────────────────────────────────────────────────────────
    def _tick(self):
        r1 = self._ray(1, self.cam1_frame)
        r2 = self._ray(2, self.cam2_frame)
        if r1 is None or r2 is None:
            return
        C1, d1 = r1
        C2, d2 = r2

        # Optional: anchor the baseline length to the tape measurement
        if self.baseline_override > 0:
            v = C2 - C1
            n = np.linalg.norm(v)
            if n > 1e-9:
                C2 = C1 + v * (self.baseline_override / n)

        # Midpoint of the common perpendicular between the two rays
        b = float(np.dot(d1, d2))
        denom = 1.0 - b * b
        if denom < 1e-6:           # rays ~parallel → depth undefined
            self.get_logger().warn(
                'Rays nearly parallel — geometric depth unreliable',
                throttle_duration_sec=5.0)
            return
        w0 = C1 - C2
        dd = float(np.dot(d1, w0))
        e  = float(np.dot(d2, w0))
        s = (b * e - dd) / denom
        t = (e - b * dd) / denom
        p1 = C1 + s * d1
        p2 = C2 + t * d2
        midpoint = 0.5 * (p1 + p2)
        gap = float(np.linalg.norm(p1 - p2))

        # ── Publish ───────────────────────────────────────────────────────
        gm = Float32(); gm.data = gap
        self.pub_gap.publish(gm)

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.world_frame
        ps.pose.position.x = float(midpoint[0])
        ps.pose.position.y = float(midpoint[1])
        ps.pose.position.z = float(midpoint[2])
        # Orientation: geometric triangulation is position-only; reuse
        # cam1's PnP orientation so the message is a complete pose.
        if self.cam1_pose_latest is not None \
                and self._fresh(self.cam1_pose_latest):
            ps.pose.orientation = self.cam1_pose_latest.pose.orientation
        else:
            ps.pose.orientation.w = 1.0
        self.pub_pose.publish(ps)

        # Distance from world origin (= cam1 lens): √(x²+y²+z²).
        dist = Float32()
        dist.data = float(np.linalg.norm(midpoint))
        self.pub_dist.publish(dist)

        # ── Scale check vs pose fusion ────────────────────────────────────
        if self.fused_latest is not None and self._fresh(self.fused_latest):
            f = self.fused_latest.pose.position
            n_fused = float(np.linalg.norm([f.x, f.y, f.z]))
            n_geo   = float(np.linalg.norm(midpoint))
            if n_fused > 1e-6:
                sc = Float32()
                sc.data = n_geo / n_fused
                self.pub_scale.publish(sc)


def main(args=None):
    rclpy.init(args=args)
    node = GeometricTriangulationNode()
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