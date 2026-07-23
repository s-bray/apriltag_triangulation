#!/usr/bin/env python3
"""
multi_apriltag_fusion_node.py — Method A generalized to N cameras (default 3).

Same pipeline as the 2-camera apriltag_triangulation_node, plus MAJORITY
VOTING: with >= 3 cameras, a camera whose world-frame estimate disagrees
with the agreeing majority is identified and dropped, instead of the blind
"fall back to cam1" of the 2-camera version.

Subscribes (per camera i in 1..n_cams):
  /cam{i}/apriltag_pose        (PoseStamped, in cam{i}'s own frame)

Publishes:
  /apriltag/triangulated_pose  (PoseStamped)  fused pose in world
  /apriltag/cam{i}_pose        (PoseStamped)  each camera in world
  /apriltag/pairwise_error_{i}{j} (Float32)   ||t_i - t_j|| per pair (m)
  /apriltag/active_cameras     (Float32)      how many cameras were fused
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
from itertools import combinations


def pose_to_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([pose.orientation.x, pose.orientation.y,
                             pose.orientation.z, pose.orientation.w]).as_matrix()
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


def matrix_to_pose(T):
    from geometry_msgs.msg import Pose
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
    q = R.from_matrix(T[:3, :3]).as_quat()
    p.orientation.x, p.orientation.y = float(q[0]), float(q[1])
    p.orientation.z, p.orientation.w = float(q[2]), float(q[3])
    return p


def average_pose_set(Ts):
    """Average a list of 4x4 poses: linear translation, hemisphere-aligned
    quaternion mean."""
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    qs = np.array([R.from_matrix(T[:3, :3]).as_quat() for T in Ts])
    for i in range(1, len(qs)):
        if np.dot(qs[0], qs[i]) < 0:
            qs[i] = -qs[i]
    q = np.mean(qs, axis=0)
    q /= np.linalg.norm(q)
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


class MultiAprilTagFusionNode(Node):

    def __init__(self):
        super().__init__('multi_apriltag_fusion_node')

        self.declare_parameter('n_cams', 3)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('max_discrepancy_m', 0.08)
        self.declare_parameter('fusion_rate_hz', 20.0)
        self.declare_parameter('max_age_sec', 1.0)

        self.n = self.get_parameter('n_cams').value
        self.world_frame = self.get_parameter('world_frame').value
        self.max_disc = self.get_parameter('max_discrepancy_m').value
        rate_hz = self.get_parameter('fusion_rate_hz').value
        self.max_age = self.get_parameter('max_age_sec').value

        self.cam_frames = [f'cam{i}_optical_frame' for i in range(1, self.n + 1)]

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest = {i: None for i in range(1, self.n + 1)}
        for i in range(1, self.n + 1):
            self.create_subscription(
                PoseStamped, f'/cam{i}/apriltag_pose',
                lambda m, idx=i: self._cb(idx, m), 10)

        self.pub_fused = self.create_publisher(
            PoseStamped, '/apriltag/triangulated_pose', 10)
        self.pub_dist = self.create_publisher(
            Float32, '/apriltag/triangulated_distance', 10)
        self.pub_cam = {
            i: self.create_publisher(
                PoseStamped, f'/apriltag/cam{i}_pose', 10)
            for i in range(1, self.n + 1)}
        self.pub_pair = {
            (i, j): self.create_publisher(
                Float32, f'/apriltag/pairwise_error_{i}{j}', 10)
            for i, j in combinations(range(1, self.n + 1), 2)}
        self.pub_active = self.create_publisher(
            Float32, '/apriltag/active_cameras', 10)

        self.create_timer(1.0 / rate_hz, self._fuse)
        self.get_logger().info(
            f'Multi-camera fusion ready ({self.n} cameras, voting enabled)')

    def _cb(self, i, msg):
        self.latest[i] = msg

    def _fresh(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return (now - t) < self.max_age

    def _to_world(self, ps):
        try:
            ps_w = self.tf_buffer.transform(
                ps, self.world_frame,
                timeout=rclpy.duration.Duration(seconds=0.1))
            return pose_to_matrix(ps_w.pose)
        except Exception as e:
            self.get_logger().warn(f'TF failed: {e}',
                                   throttle_duration_sec=5.0)
            return None

    def _stamped(self, T):
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.world_frame
        ps.pose = matrix_to_pose(T)
        return ps

    def _fuse(self):
        # World-frame pose per currently-fresh camera
        world = {}
        for i in range(1, self.n + 1):
            m = self.latest[i]
            if m is not None and self._fresh(m):
                T = self._to_world(m)
                if T is not None:
                    world[i] = T
                    self.pub_cam[i].publish(self._stamped(T))

        if not world:
            return

        # Pairwise errors (published for every visible pair)
        pd = {}
        for i, j in combinations(sorted(world), 2):
            d = float(np.linalg.norm(world[i][:3, 3] - world[j][:3, 3]))
            pd[(i, j)] = d
            if (i, j) in self.pub_pair:
                msg = Float32(); msg.data = d
                self.pub_pair[(i, j)].publish(msg)

        # ── Voting: keep the largest subset that is mutually consistent ──
        cams = sorted(world)
        best = [cams[0]]                       # worst case: lowest-index cam
        for r in range(len(cams), 0, -1):
            found = None
            for subset in combinations(cams, r):
                ok = all(pd[(a, b)] < self.max_disc
                         for a, b in combinations(subset, 2))
                if ok:
                    found = list(subset)
                    break
            if found:
                best = found
                break

        dropped = set(cams) - set(best)
        if dropped:
            self.get_logger().warn(
                f'Voting dropped camera(s) {sorted(dropped)} '
                f'(disagree > {self.max_disc} m)',
                throttle_duration_sec=2.0)

        T_fused = average_pose_set([world[i] for i in best])
        self.pub_fused.publish(self._stamped(T_fused))

        # Distance from world origin (= cam1 lens): √(x²+y²+z²).
        dist = Float32()
        dist.data = float(np.linalg.norm(T_fused[:3, 3]))
        self.pub_dist.publish(dist)

        a = Float32(); a.data = float(len(best))
        self.pub_active.publish(a)


def main(args=None):
    rclpy.init(args=args)
    node = MultiAprilTagFusionNode()
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
