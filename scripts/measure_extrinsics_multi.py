#!/usr/bin/env python3
"""
measure_extrinsics_multi.py — calibrate cam2 AND cam3 relative to cam1
in a single run, using one stationary AprilTag.

For each secondary camera X:
    T_cam1_camX = T_cam1_tag @ inv(T_camX_tag)

Samples for pair (1,X) are collected whenever cam1 AND camX are both
fresh — camX pairs fill independently, so the tag doesn't need to be in
all three views at once (though a spot visible to all three is fastest).

All the v3 safeguards carry over per pair: freshness pairing, spread
gate, MAD outlier rejection, quaternion hemisphere averaging, and the
scale diagnostic against known_baseline (checked on the cam1-cam2 pair).

Usage:
    ros2 run apriltag_triangulation measure_extrinsics_multi \
        --ros-args -p n_samples:=200 \
        -p known_baseline:=0.59 -p current_tag_size:=0.15
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np


def pose_to_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([pose.orientation.x, pose.orientation.y,
                             pose.orientation.z, pose.orientation.w]).as_matrix()
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


def robust_average(T_list, mad_k=3.0):
    """MAD-filter on baseline length, then average. Returns
    (t_mean, t_std, q_mean, baseline, n_rejected, spread)."""
    translations = np.array([T[:3, 3] for T in T_list])
    rotations = np.array([R.from_matrix(T[:3, :3]).as_quat() for T in T_list])
    baselines = np.linalg.norm(translations, axis=1)
    spread = float(np.max(baselines) - np.min(baselines))

    median = np.median(baselines)
    mad = max(np.median(np.abs(baselines - median)), 1e-8)
    mask = np.abs(baselines - median) < mad_k * 1.4826 * mad

    translations = translations[mask]
    rotations = rotations[mask]

    t_mean = np.mean(translations, axis=0)
    t_std = np.std(translations, axis=0)

    for i in range(1, len(rotations)):
        if np.dot(rotations[0], rotations[i]) < 0:
            rotations[i] = -rotations[i]
    q_mean = np.mean(rotations, axis=0)
    q_mean /= np.linalg.norm(q_mean)

    return (t_mean, t_std, q_mean, float(np.linalg.norm(t_mean)),
            int(np.sum(~mask)), spread)


class MultiExtrinsicMeasurer(Node):

    def __init__(self):
        super().__init__('multi_extrinsic_calibrator')

        self.declare_parameter('n_samples', 200)
        self.declare_parameter('known_baseline', 0.0)   # tape cam1<->cam2 (m)
        self.declare_parameter('current_tag_size', 0.1)
        self.declare_parameter('max_age_sec', 1.0)
        self.declare_parameter('mad_k', 3.0)

        self.n_samples  = self.get_parameter('n_samples').value
        self.known_bl   = self.get_parameter('known_baseline').value
        self.tag_size   = self.get_parameter('current_tag_size').value
        self.max_age    = self.get_parameter('max_age_sec').value
        self.mad_k      = self.get_parameter('mad_k').value

        self.latest = {1: None, 2: None, 3: None}
        self.samples = {2: [], 3: []}   # pair (1,X) → list of T_cam1_camX

        for i in (1, 2, 3):
            self.create_subscription(
                PoseStamped, f'/cam{i}/apriltag_pose',
                lambda m, idx=i: self._cb(idx, m), 10)

        self.create_timer(0.1, self._collect)
        self.get_logger().info(
            f'Multi-extrinsic calibration: {self.n_samples} samples per pair '
            '(1-2 and 1-3). Tag rigid and still; each pair fills whenever '
            'cam1 and that camera both see it.')

    def _cb(self, i, msg):
        self.latest[i] = msg

    def _fresh(self, msg):
        if msg is None:
            return False
        now = self.get_clock().now().nanoseconds * 1e-9
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return (now - t) < self.max_age

    def _collect(self):
        if not self._fresh(self.latest[1]):
            self.get_logger().info(
                'Waiting for cam1 (reference camera) to see the tag...',
                throttle_duration_sec=3.0)
            return

        T1 = pose_to_matrix(self.latest[1].pose)
        progressed = False

        for X in (2, 3):
            if len(self.samples[X]) >= self.n_samples:
                continue
            if self._fresh(self.latest[X]):
                TX = pose_to_matrix(self.latest[X].pose)
                self.samples[X].append(T1 @ np.linalg.inv(TX))
                progressed = True

        if progressed:
            self.get_logger().info(
                f'pair 1-2: {len(self.samples[2])}/{self.n_samples}   '
                f'pair 1-3: {len(self.samples[3])}/{self.n_samples}')

        if all(len(self.samples[X]) >= self.n_samples for X in (2, 3)):
            self._compute()
            rclpy.shutdown()

    def _compute(self):
        results = {}
        print('\n' + '=' * 60)
        print('MULTI-CAMERA EXTRINSIC RESULTS')
        print('=' * 60)

        for X in (2, 3):
            t, ts, q, bl, nrej, spread = robust_average(
                self.samples[X], self.mad_k)
            results[X] = (t, q)
            print(f'\n── cam{X} relative to cam1 ─────────────────────────')
            print(f'  translation : {t[0]:+.6f}  {t[1]:+.6f}  {t[2]:+.6f}')
            print(f'  std         : {ts[0]:.6f}  {ts[1]:.6f}  {ts[2]:.6f}')
            print(f'  quaternion  : {q[0]:+.6f}  {q[1]:+.6f}  '
                  f'{q[2]:+.6f}  {q[3]:+.6f}')
            print(f'  baseline    : {bl:.6f} m   '
                  f'(rejected {nrej}, raw spread {spread*1000:.1f} mm)')
            if spread > 0.02:
                print('  !! spread > 2 cm on a static scene — result NOT')
                print('  !! trustworthy for this pair; find the cause '
                  '(movement /')
                print('  !! shared TF frames / ambiguity flips) and re-run.')

        # Scale diagnostic on the 1-2 pair
        if self.known_bl > 0:
            bl12 = float(np.linalg.norm(results[2][0]))
            err = (bl12 - self.known_bl) / self.known_bl * 100.0
            print(f'\nSCALE DIAGNOSTIC (pair 1-2): measured {bl12:.4f} m vs '
                  f'tape {self.known_bl:.4f} m → {err:+.2f}%')
            if abs(err) >= 1.5:
                print(f'  → fx·L product off by ~{err:+.1f}%. tag_size '
                      f'caliper-verified ⇒ intrinsics; else try tag_size '
                      f'= {self.tag_size * self.known_bl / bl12:.4f} m.')
                print('  → Do NOT rescale these extrinsics; fix the cause '
                      'and re-run.')

        t2, q2 = results[2]
        t3, q3 = results[3]
        print('\nLaunch command:')
        print(f'''
ros2 launch apriltag_triangulation triple_apriltag_triangulation.launch.py \\
    tag_size:={self.tag_size:.4f} tag_id:=0 \\
    cam2_tx:='{t2[0]:.6f}' cam2_ty:='{t2[1]:.6f}' cam2_tz:='{t2[2]:.6f}' \\
    cam2_qx:='{q2[0]:.6f}' cam2_qy:='{q2[1]:.6f}' \\
    cam2_qz:='{q2[2]:.6f}' cam2_qw:='{q2[3]:.6f}' \\
    cam3_tx:='{t3[0]:.6f}' cam3_ty:='{t3[1]:.6f}' cam3_tz:='{t3[2]:.6f}' \\
    cam3_qx:='{q3[0]:.6f}' cam3_qy:='{q3[1]:.6f}' \\
    cam3_qz:='{q3[2]:.6f}' cam3_qw:='{q3[3]:.6f}'
''')
        print('(add your cam*_device and cam*_calib arguments)')
        print('=' * 60)


def main(args=None):
    rclpy.init(args=args)
    node = MultiExtrinsicMeasurer()
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
