#!/usr/bin/env python3
"""
measure_extrinsics_multi.py — v2

Calibrates cam1→cam2 AND cam1→cam3 in one run, where each pair is
processed EXACTLY like the proven 2-camera script: same freshness
pairing, same pre-filter statistics printout, same >2 cm spread gate,
same MAD outlier rejection report, and the same fx·L scale diagnostic —
now per pair, with a tape-measured known baseline for each.

    T_cam1_camX = T_cam1_tag @ inv(T_camX_tag)

Pairs fill independently: a (1,X) sample is taken whenever cam1 and camX
are BOTH fresh, so the tag does not need to be in all three views at
once (though a triple-visible spot fills fastest).

Usage:
    ros2 run apriltag_triangulation measure_extrinsics_multi \
        --ros-args -p n_samples:=200 \
        -p known_baseline_12:=0.59 \
        -p known_baseline_13:=0.72 \
        -p current_tag_size:=0.15
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


class MultiExtrinsicMeasurer(Node):

    def __init__(self):
        super().__init__('multi_extrinsic_calibrator')

        self.declare_parameter('n_samples', 200)
        self.declare_parameter('known_baseline_12', 0.0)  # tape cam1<->cam2 (m)
        self.declare_parameter('known_baseline_13', 0.0)  # tape cam1<->cam3 (m)
        self.declare_parameter('current_tag_size', 0.1)
        self.declare_parameter('max_age_sec', 1.0)
        self.declare_parameter('mad_k', 3.0)

        self.n_samples = self.get_parameter('n_samples').value
        self.known_bl = {
            2: self.get_parameter('known_baseline_12').value,
            3: self.get_parameter('known_baseline_13').value,
        }
        self.tag_size = self.get_parameter('current_tag_size').value
        self.max_age  = self.get_parameter('max_age_sec').value
        self.mad_k    = self.get_parameter('mad_k').value

        self.latest = {1: None, 2: None, 3: None}
        self.samples = {2: [], 3: []}   # pair (1,X) → list of T_cam1_camX

        for i in (1, 2, 3):
            self.create_subscription(
                PoseStamped, f'/cam{i}/apriltag_pose',
                lambda m, idx=i: self._cb(idx, m), 10)

        self.create_timer(0.1, self._collect)
        self.get_logger().info(
            '\n' + '=' * 60 +
            '\nMulti-camera Extrinsic Calibration (v2 — per-pair '
            'diagnostics)\n' + '=' * 60 +
            f'\n  n_samples/pair    : {self.n_samples}' +
            f'\n  known_baseline_12 : {self.known_bl[2]:.4f} m '
            f'({"diagnostic ON" if self.known_bl[2] > 0 else "off"})' +
            f'\n  known_baseline_13 : {self.known_bl[3]:.4f} m '
            f'({"diagnostic ON" if self.known_bl[3] > 0 else "off"})' +
            f'\n  current_tag_size  : {self.tag_size:.4f} m '
            '(must match live apriltag nodes)' +
            f'\n  max_age_sec       : {self.max_age:.2f} s' +
            '\nHold the tag STILL on a rigid mount, visible to all '
            'cameras if possible.')

    # ── Collection ────────────────────────────────────────────────────────
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
            results = {}
            for X in (2, 3):
                results[X] = self._analyze_pair(X)
            self._print_launch(results)
            rclpy.shutdown()

    # ── Per-pair analysis: EXACT replica of the 2-camera script ──────────
    def _analyze_pair(self, X):
        T_list = self.samples[X]
        translations = np.array([T[:3, 3] for T in T_list])
        rotations = np.array([R.from_matrix(T[:3, :3]).as_quat()
                              for T in T_list])
        baselines = np.linalg.norm(translations, axis=1)

        print('\n' + '=' * 60)
        print(f'PAIR cam1–cam{X}')
        print('=' * 60)
        print('Baseline statistics BEFORE outlier rejection')
        print('-' * 60)
        print(f'  min    : {np.min(baselines):.6f} m')
        print(f'  max    : {np.max(baselines):.6f} m')
        print(f'  mean   : {np.mean(baselines):.6f} m')
        print(f'  median : {np.median(baselines):.6f} m')
        print(f'  std    : {np.std(baselines):.6f} m')

        # ── Spread gate ──────────────────────────────────────────────────
        spread = float(np.max(baselines) - np.min(baselines))
        if spread > 0.02:
            print('\n' + '!' * 60)
            print(f'  WARNING: pair 1-{X} baseline spread is '
                  f'{spread*100:.1f} cm on a')
            print('  supposedly static scene (expected: a few millimetres).')
            print('  The averaged result below is NOT trustworthy. Likely causes:')
            print('    - tag or a camera moved during collection')
            print('    - both cameras publishing the same TF tag frame')
            print('      (must be unique, e.g. tag0_cam1 / tag0_cam2 / tag0_cam3)')
            print('    - pose-ambiguity flips (tag too frontal — angle it 15-25°)')
            print('  Fix the cause and re-run before using these numbers.')
            print('!' * 60)

        # ── MAD outlier rejection ────────────────────────────────────────
        median = np.median(baselines)
        mad = max(np.median(np.abs(baselines - median)), 1e-8)
        threshold = self.mad_k * 1.4826 * mad
        mask = np.abs(baselines - median) < threshold

        print('\nOutlier rejection (MAD-based)')
        print('-' * 60)
        print(f'  rejected       : {np.sum(~mask)}')
        print(f'  accepted       : {np.sum(mask)}')
        print(f'  accepted range : {median - threshold:.6f} to '
              f'{median + threshold:.6f} m')

        translations = translations[mask]
        rotations = rotations[mask]

        if len(translations) < 5:
            print(f'\n[ERROR] pair 1-{X}: too few samples survived outlier '
                  'rejection. Re-run with the tag held more still.')
            return None

        t_mean = np.mean(translations, axis=0)
        t_std = np.std(translations, axis=0)
        measured_baseline = float(np.linalg.norm(t_mean))

        q_ref = rotations[0].copy()
        for i in range(len(rotations)):
            if np.dot(q_ref, rotations[i]) < 0:
                rotations[i] = -rotations[i]
        q_mean = np.mean(rotations, axis=0)
        q_mean /= np.linalg.norm(q_mean)

        # ── Scale diagnostic (never applied to the result) ───────────────
        print('\nSCALE DIAGNOSTIC')
        print('-' * 60)
        kb = self.known_bl[X]
        if kb > 0.0:
            error_pct = (measured_baseline - kb) / kb * 100.0
            implied_tag_size = self.tag_size * (kb / measured_baseline)

            print(f'  Measured baseline              : {measured_baseline:.6f} m')
            print(f'  Known (tape-measured) baseline : {kb:.6f} m')
            print(f'  Error                          : {error_pct:+.2f} %')
            print(f'  Currently configured tag_size  : {self.tag_size:.4f} m')

            if abs(error_pct) < 1.5:
                print('\n  → Within ~1.5%. System scale looks correct.')
            else:
                scale = kb / measured_baseline
                print(f'\n  → SYSTEM SCALE ERROR of {error_pct:+.1f}% detected.')
                print(f'    Correction scale (this setup only):')
                print(f'      true ≈ {scale:.4f} × system_output')
                print(f'      equivalently: depth_scale param = {1.0/scale:.4f}')
                print('    CAUTION: only guaranteed near the tag position used')
                print('    in THIS run; if the cause is intrinsics (fx), the')
                print('    factor varies across the workspace.')
                print('    The script measures only the fx·L product; it cannot')
                print('    tell which factor is wrong:')
                print(f'      (a) tag_size unverified → try {implied_tag_size:.4f} m')
                print(f'      (b) tag_size caliper-verified → intrinsics (fx)')
                print(f'          of cam1 and/or cam{X} off by ~{(scale-1)*100:+.1f}% —')
                print('          recalibrate per camera, check calib resolution')
                print('          matches the stream.')
                print('    Hint: error differing between pair 1-2 and pair 1-3')
                print('    → per-camera intrinsics (b), since tag_size errors')
                print('    are identical for every pair.')
                print('  → Do NOT force-rescale this extrinsic to match the')
                print('    known baseline (breaks runtime consistency).')
                print('  → Fix the cause upstream, relaunch, re-run FRESH.')
        else:
            print(f'  known_baseline_1{X} not provided (or 0) — skipping.')
            print(f'  Pass -p known_baseline_1{X}:=<tape_m> to check for a')
            print('  tag_size / intrinsics scale error on this pair.')

        # ── Pair result ──────────────────────────────────────────────────
        print(f'''
RESULT pair 1-{X}  (uncorrected — use as-is)
Translation cam{X} relative to cam1:
  x = {t_mean[0]:.6f}
  y = {t_mean[1]:.6f}
  z = {t_mean[2]:.6f}
Translation std (aim for < 0.01 m):
  x = {t_std[0]:.6f}
  y = {t_std[1]:.6f}
  z = {t_std[2]:.6f}
Quaternion:
  qx = {q_mean[0]:.6f}
  qy = {q_mean[1]:.6f}
  qz = {q_mean[2]:.6f}
  qw = {q_mean[3]:.6f}
Measured baseline: {measured_baseline:.6f} m''')

        return (t_mean, q_mean)

    # ── Combined launch command ──────────────────────────────────────────
    def _print_launch(self, results):
        if results.get(2) is None or results.get(3) is None:
            print('\nOne or more pairs failed — fix and re-run before '
                  'launching.')
            return
        t2, q2 = results[2]
        t3, q3 = results[3]
        print('\n' + '=' * 60)
        print('Launch command (uses current_tag_size — recalibrate if '
              'changed):')
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