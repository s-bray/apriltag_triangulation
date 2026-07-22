#!/usr/bin/env python3
"""
measure_cam2_extrinsic_apriltag.py  — v2

Dual-camera extrinsic calibration using a single stationary AprilTag,
subscribing directly to /cam1/apriltag_pose and /cam2/apriltag_pose.

    T_cam1_cam2 = T_cam1_tag @ inv(T_cam2_tag)

Changes vs the v1 script:

  1. FRESHNESS CHECK — every sample now requires BOTH camera poses to be
     recent (age < max_age_sec since last detection). The adapter node
     only publishes /camX/apriltag_pose while the tag is actually visible,
     so if a camera loses the tag for a few seconds, its "latest" stored
     message just gets older and older. The old version paired that stale
     message against the other camera's fresh one on every 0.1 s tick,
     silently collecting bad samples the whole time. This version skips
     the tick instead and tells you which camera is stale.

  2. known_baseline IS NOW DIAGNOSTIC ONLY — it no longer rescales the
     translation used in the final result. A wrong tag_size scales EVERY
     live detection from BOTH cameras by the same factor, so cam1's and
     cam2's world estimates stay mutually consistent even though both are
     wrong in absolute terms — like a map with the wrong scale bar but
     correct proportions. Forcibly rescaling only the extrinsic breaks
     that consistency: the runtime discrepancy between the two cameras
     gets WORSE, not better, and it grows with distance from the tag.
     Instead, this script compares the measured baseline against
     known_baseline and tells you what tag_size to try instead, then
     asks you to redo the calibration from scratch with the corrected
     tag_size already applied to the live apriltag nodes.

Usage:
    ros2 run apriltag_triangulation measure_cam2_extrinsic_apriltag \
        --ros-args \
        -p n_samples:=200 \
        -p known_baseline:=0.90 \
        -p current_tag_size:=0.139
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np


def pose_to_matrix(pose):
    t = np.array([pose.position.x, pose.position.y, pose.position.z])
    q = [pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


class ExtrinsicMeasurer(Node):

    def __init__(self):
        super().__init__('apriltag_extrinsic_calibrator')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('n_samples', 200)
        self.declare_parameter('known_baseline', 0.0)      # m, 0 = skip diagnostic
        self.declare_parameter('current_tag_size', 0.1)    # m, must match live tags.yaml
        self.declare_parameter('max_age_sec', 0.2)         # reject stale pose pairs
        self.declare_parameter('mad_k', 3.0)                # outlier rejection strictness

        self.n_samples         = self.get_parameter('n_samples').value
        self.known_baseline    = self.get_parameter('known_baseline').value
        self.current_tag_size  = self.get_parameter('current_tag_size').value
        self.max_age_sec       = self.get_parameter('max_age_sec').value
        self.mad_k             = self.get_parameter('mad_k').value

        self.samples = []
        self.latest_cam1 = None
        self.latest_cam2 = None

        self.create_subscription(PoseStamped, '/cam1/apriltag_pose', self._cb1, 10)
        self.create_subscription(PoseStamped, '/cam2/apriltag_pose', self._cb2, 10)
        self.create_timer(0.1, self._collect)

        self.get_logger().info(
            '\n' + '=' * 60 +
            '\nAprilTag Stereo Extrinsic Calibration (v2)\n' + '=' * 60 +
            f'\n  n_samples        : {self.n_samples}' +
            f'\n  known_baseline   : {self.known_baseline:.4f} m '
            f'({"diagnostic ON" if self.known_baseline > 0 else "off"})' +
            f'\n  current_tag_size : {self.current_tag_size:.4f} m '
            '(must match what the live apriltag nodes use right now)' +
            f'\n  max_age_sec      : {self.max_age_sec:.2f} s' +
            '\nHold the tag STILL — ideally on a rigid mount, not by hand —'
            '\nwhere BOTH cameras can see it clearly.'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────
    def _cb1(self, msg): self.latest_cam1 = msg
    def _cb2(self, msg): self.latest_cam2 = msg

    def _is_fresh(self, msg: PoseStamped) -> bool:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        msg_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return (now_sec - msg_sec) < self.max_age_sec

    # ── Sample collection ────────────────────────────────────────────────
    def _collect(self):
        c1, c2 = self.latest_cam1, self.latest_cam2

        if c1 is None or c2 is None:
            self.get_logger().info(
                'Waiting for both cameras to publish at least once...',
                throttle_duration_sec=3.0)
            return

        fresh1, fresh2 = self._is_fresh(c1), self._is_fresh(c2)
        if not (fresh1 and fresh2):
            missing = []
            if not fresh1: missing.append('cam1 (stale / tag not visible)')
            if not fresh2: missing.append('cam2 (stale / tag not visible)')
            self.get_logger().info(
                f'Skipping tick — {", ".join(missing)}',
                throttle_duration_sec=2.0)
            return

        T_cam1_tag  = pose_to_matrix(c1.pose)
        T_cam2_tag  = pose_to_matrix(c2.pose)
        T_cam1_cam2 = T_cam1_tag @ np.linalg.inv(T_cam2_tag)

        self.samples.append(T_cam1_cam2)
        n = len(self.samples)
        self.get_logger().info(f'Sample {n}/{self.n_samples}')

        if n >= self.n_samples:
            self._compute_result()
            rclpy.shutdown()

    # ── Result computation ───────────────────────────────────────────────
    def _compute_result(self):
        translations = np.array([T[:3, 3] for T in self.samples])
        rotations    = np.array([R.from_matrix(T[:3, :3]).as_quat()
                                  for T in self.samples])
        baselines    = np.linalg.norm(translations, axis=1)

        print('\n' + '=' * 60)
        print('Baseline statistics BEFORE outlier rejection')
        print('-' * 60)
        print(f'  min    : {np.min(baselines):.6f} m')
        print(f'  max    : {np.max(baselines):.6f} m')
        print(f'  mean   : {np.mean(baselines):.6f} m')
        print(f'  median : {np.median(baselines):.6f} m')
        print(f'  std    : {np.std(baselines):.6f} m')

        # ── Data-quality gate ────────────────────────────────────────────
        # MAD rejection only removes outliers relative to the bulk; it
        # cannot save a run where the bulk itself is smeared (e.g. TF
        # frame conflicts or pose-ambiguity flips create a bimodal
        # distribution and MAD passes everything). On a static scene the
        # spread should be millimetres — refuse to bless anything wider.
        spread = float(np.max(baselines) - np.min(baselines))
        if spread > 0.02:
            print('\n' + '!' * 60)
            print(f'  WARNING: baseline spread is {spread*100:.1f} cm on a')
            print('  supposedly static scene (expected: a few millimetres).')
            print('  The averaged result below is NOT trustworthy. Likely causes:')
            print('    - tag or a camera moved during collection')
            print('    - both cameras publishing the same TF tag frame')
            print('      (must be unique, e.g. tag0_cam1 / tag0_cam2)')
            print('    - pose-ambiguity flips (tag too frontal — angle it 15-25°)')
            print('  Fix the cause and re-run before using these numbers.')
            print('!' * 60)

        # ── MAD-based outlier rejection ───────────────────────────────────
        median = np.median(baselines)
        mad = np.median(np.abs(baselines - median))
        if mad < 1e-8:
            mad = 1e-8
        threshold = self.mad_k * 1.4826 * mad
        mask = np.abs(baselines - median) < threshold

        print('\nOutlier rejection (MAD-based)')
        print('-' * 60)
        print(f'  rejected       : {np.sum(~mask)}')
        print(f'  accepted       : {np.sum(mask)}')
        print(f'  accepted range : {median - threshold:.6f} to '
              f'{median + threshold:.6f} m')

        translations = translations[mask]
        rotations    = rotations[mask]

        if len(translations) < 5:
            print('\n[ERROR] Too few samples survived outlier rejection.')
            print('Re-run with the tag held more still, on a rigid mount.')
            return

        # ── Translation average — NATURAL, UNCORRECTED ───────────────────
        t_mean = np.mean(translations, axis=0)
        t_std  = np.std(translations, axis=0)
        measured_baseline = float(np.linalg.norm(t_mean))

        # ── Quaternion average ────────────────────────────────────────────
        q_ref = rotations[0].copy()
        for i in range(len(rotations)):
            if np.dot(q_ref, rotations[i]) < 0:
                rotations[i] = -rotations[i]
        q_mean = np.mean(rotations, axis=0)
        q_mean /= np.linalg.norm(q_mean)

        # ── Diagnostic: tag_size scale check (NEVER applied to t_mean) ────
        print('\n' + '=' * 60)
        print('TAG SIZE DIAGNOSTIC')
        print('=' * 60)
        if self.known_baseline > 0.0:
            error_pct = ((measured_baseline - self.known_baseline)
                         / self.known_baseline * 100.0)
            implied_tag_size = (self.current_tag_size
                                * (self.known_baseline / measured_baseline))

            print(f'  Measured baseline             : {measured_baseline:.6f} m')
            print(f'  Known (tape-measured) baseline : {self.known_baseline:.6f} m')
            print(f'  Error                          : {error_pct:+.2f} %')
            print(f'  Currently configured tag_size  : {self.current_tag_size:.4f} m')

            if abs(error_pct) < 1.5:
                print('\n  → Within ~1.5%. System scale looks correct — no change needed.')
            else:
                scale = self.known_baseline / measured_baseline
                print(f'\n  → SYSTEM SCALE ERROR of {error_pct:+.1f}% detected.')
                print(f'    Correction scale (this setup only):')
                print(f'      true ≈ {scale:.4f} × system_output')
                print(f'      equivalently: depth_scale param = '
                      f'{1.0/scale:.4f}  (= measured/true, for cam1/cam2_depth_scale)')
                print('    CAUTION: this factor is only guaranteed near the tag')
                print('    position used in THIS run. If the cause is intrinsics')
                print('    (fx) rather than tag_size, the effective factor varies')
                print('    across the workspace — verify at multiple positions')
                print('    before trusting it as a blanket correction.')
                print('    The script can only measure the fx·L product (focal')
                print('    length × tag size); it CANNOT tell which factor is wrong:')
                print(f'      (a) if tag_size is unverified: try tag_size = '
                      f'{implied_tag_size:.4f} m')
                print(f'      (b) if tag_size is caliper-verified: your camera')
                print(f'          INTRINSICS (fx) are off by ~{(scale-1)*100:+.1f}% —')
                print('          recalibrate each camera separately (checkerboard),')
                print('          check calibration resolution matches the stream,')
                print('          and ensure apriltag_ros receives rectified images.')
                print('    Hint: if the error % changes sign or magnitude between')
                print('    different camera geometries, it is (b), not (a) — a')
                print('    tag_size error is identical in every setup.')
                print('  → Either way, do NOT force-rescale this extrinsic to match')
                print('    known_baseline: the same scale error is baked into every')
                print('    live detection too, so extrinsic and live data currently')
                print('    agree with each other. Rescaling only the extrinsic breaks')
                print('    that agreement (runtime discrepancy grows with distance).')
                print('  → Fix the actual cause upstream, relaunch, then re-run this')
                print('    calibration FRESH. The measured baseline should then land')
                print('    on known_baseline by itself, with no correction applied.')
        else:
            print('  known_baseline not provided (or 0) — skipping diagnostic.')
            print('  Pass -p known_baseline:=<tape_measured_distance_m> next time')
            print('  to check for a tag_size scale error.')

        # ── Final result — ALWAYS the uncorrected / natural values ────────
        print('\n' + '=' * 60)
        print('FINAL AprilTag Stereo Extrinsic  (uncorrected — use as-is)')
        print('=' * 60)
        print(f'''
Translation cam2 relative to cam1:
  x = {t_mean[0]:.6f}
  y = {t_mean[1]:.6f}
  z = {t_mean[2]:.6f}

Translation std (calibration quality — aim for < 0.01 m):
  x = {t_std[0]:.6f}
  y = {t_std[1]:.6f}
  z = {t_std[2]:.6f}

Quaternion:
  qx = {q_mean[0]:.6f}
  qy = {q_mean[1]:.6f}
  qz = {q_mean[2]:.6f}
  qw = {q_mean[3]:.6f}

Measured baseline (uncorrected): {measured_baseline:.6f} m
''')

        print('Launch command (uses current_tag_size — recalibrate if you change it):')
        print(f'''
ros2 launch apriltag_triangulation dual_apriltag_triangulation.launch.py \\
    cam1_device:=/dev/video0 \\
    cam2_device:=/dev/video2 \\
    cam1_calib:=file:///home/ros/ws/src/camera_calibrations/camera_calib/ost.yaml \\
    cam2_calib:=file:///home/ros/ws/src/camera_calibrations/camera_calib2/ost.yaml \\
    tag_size:={self.current_tag_size:.4f} \\
    tag_id:=0 \\
    cam2_tx:='{t_mean[0]:.6f}' \\
    cam2_ty:='{t_mean[1]:.6f}' \\
    cam2_tz:='{t_mean[2]:.6f}' \\
    cam2_qx:='{q_mean[0]:.6f}' \\
    cam2_qy:='{q_mean[1]:.6f}' \\
    cam2_qz:='{q_mean[2]:.6f}' \\
    cam2_qw:='{q_mean[3]:.6f}'
''')
        print('=' * 60)


def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicMeasurer()
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