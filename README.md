# AprilTag Dual-Camera Ground-Truth Tracking

A ROS2 Jazzy package that tracks a single AprilTag with **two webcams** and
fuses their observations into an accurate world-frame pose. Built as a
low-cost ground-truth system (a poor man's OptiTrack) for evaluating robot
localisation and navigation.

Two independent methods run side by side:

| Method | Node | Depth comes from | Output topic |
|---|---|---|---|
| **Pose fusion** (PnP average) | `apriltag_triangulation_node` | tag's apparent size (fx·L) | `/apriltag/triangulated_pose` |
| **Geometric triangulation** (ray intersection) | `geometric_triangulation_node` | camera baseline geometry | `/apriltag/geometric_pose` |

Having both is deliberate: they fail differently, so comparing them measures
the system's own error live.

---

## Table of Contents

1. [System architecture](#1-system-architecture)
2. [Coordinate frames](#2-coordinate-frames)
3. [The files and what each does](#3-the-files-and-what-each-does)
4. [Method A — Pose fusion, step by step](#4-method-a--pose-fusion-step-by-step)
5. [Method B — Geometric triangulation, step by step](#5-method-b--geometric-triangulation-step-by-step)
6. [Why two methods: the error model](#6-why-two-methods-the-error-model)
7. [Calibration workflow](#7-calibration-workflow)
8. [Every ROS2 topic explained](#8-every-ros2-topic-explained)
9. [Launch and parameters](#9-launch-and-parameters)
10. [Known pitfalls (learned the hard way)](#10-known-pitfalls-learned-the-hard-way)

---

## 1. System architecture

```
 /dev/video0                                /dev/video2
     │                                          │
 ┌───▼──────┐                             ┌─────▼────┐
 │ usb_cam  │  intrinsics from            │ usb_cam  │
 │  (cam1)  │  cam1_calib / cam2_calib    │  (cam2)  │
 └───┬──────┘                             └─────┬────┘
     │ /cam1/image_raw + camera_info            │ /cam2/image_raw + camera_info
 ┌───▼───────────┐                        ┌─────▼─────────┐
 │ apriltag_node │ detects tag corners,   │ apriltag_node │
 │   (cam1)      │ solves PnP, publishes  │   (cam2)      │
 └───┬───────────┘ TF: cam→tag + a        └─────┬─────────┘
     │             detections topic             │
     │ TF: cam1_optical_frame → tag0_cam1       │ TF: cam2_optical_frame → tag0_cam2
 ┌───▼───────────┐                        ┌─────▼─────────┐
 │ adapter node  │ TF → PoseStamped       │ adapter node  │
 │   (cam1)      │ bridge                 │   (cam2)      │
 └───┬───────────┘                        └─────┬─────────┘
     │ /cam1/apriltag_pose                      │ /cam2/apriltag_pose
     └──────────────────┬───────────────────────┘
                        │
      ┌─────────────────▼──────────────────┐      ┌────────────────────────────┐
      │  apriltag_triangulation_node       │      │ geometric_triangulation_   │
      │  (Method A: pose fusion)           │      │ node (Method B: rays)      │
      │  → /apriltag/triangulated_pose     │      │ reads centre PIXELS from   │
      └────────────────────────────────────┘      │ /camX/apriltag/detections  │
                                                  │ → /apriltag/geometric_pose │
                                                  └────────────────────────────┘
```

Static transforms (from the launch arguments) tie everything together:
`world → cam1_optical_frame` (identity: **cam1 IS the world origin**) and
`world → cam2_optical_frame` (the **extrinsic** you calibrate).

---

## 2. Coordinate frames

| Frame | What it is |
|---|---|
| `world` | Fixed reference frame. Defined as identical to cam1's optical frame. All final outputs are expressed here. |
| `cam1_optical_frame` | Cam1's lens coordinate system. Optical convention: **+Z forward** (into the scene), +X right, +Y down. |
| `cam2_optical_frame` | Same convention, located at cam2. Its pose relative to world is the **extrinsic** (`cam2_tx … cam2_qw`). |
| `tag0_cam1` | The tag as detected **by cam1**, child of cam1's frame. |
| `tag0_cam2` | The tag as detected **by cam2**, child of cam2's frame. |

The tag frames are deliberately **unique per camera**. If both cameras
published the same frame name (`tag0`), its TF parent would flip-flop
between the two cameras many times per second, and TF lookups would then
silently route through `world` — dragging the static launch extrinsic into
supposedly "raw" measurements. This corrupted our early calibrations; see
[Pitfalls](#10-known-pitfalls-learned-the-hard-way).

Tag frame convention: origin at the tag centre, +Z out of the tag face.

---

## 3. The files and what each does

```
apriltag_triangulation/
├── apriltag_triangulation/
│   ├── apriltag_adapter_node.py          # TF → topic bridge (one per camera)
│   ├── apriltag_triangulation_node.py    # Method A: pose fusion
│   ├── geometric_triangulation_node.py   # Method B: ray intersection
│   └── measure_cam2_extrinsic_apriltag.py# extrinsic calibration tool (v3)
├── launch/
│   └── dual_apriltag_triangulation.launch.py
├── config/
│   └── tags.yaml                         # apriltag_ros detector settings
├── package.xml / setup.py / setup.cfg / resource/
```

### `apriltag_adapter_node.py` (runs twice, once per camera)

`apriltag_ros` publishes detections into **TF**, not a topic. This node
bridges that gap for one camera:

1. Watches `/camX/apriltag/detections` and marks the tag "visible" only if
   its `decision_margin` (detector confidence, 0–255) passes
   `min_decision_margin`.
2. When visible, looks up `camX_optical_frame → tag0_camX` in TF — the tag's
   raw pose **in that camera's own frame**, untouched by any extrinsic.
3. Publishes it as `/camX/apriltag_pose` (PoseStamped) plus a
   `/camX/apriltag_detected` (Bool) flag.
4. Optionally divides the position by `depth_scale` — a per-camera range
   correction hook (default 1.0 = off).

### `apriltag_triangulation_node.py` — Method A

Fuses the two per-camera poses into one world-frame pose. Details in
[section 4](#4-method-a--pose-fusion-step-by-step).

### `geometric_triangulation_node.py` — Method B

Ignores the PnP poses. Takes each camera's detected **centre pixel**,
back-projects viewing rays, intersects them. Details in
[section 5](#5-method-b--geometric-triangulation-step-by-step).

### `measure_cam2_extrinsic_apriltag.py` — extrinsic calibration

Computes the cam1↔cam2 transform from simultaneous tag observations.
Details in [section 7](#7-calibration-workflow).

---

## 4. Method A — Pose fusion, step by step

**Inputs:** `/cam1/apriltag_pose`, `/cam2/apriltag_pose` (each already a
full 6-DoF tag pose in its own camera frame, solved by apriltag_ros's PnP).

**Where each input's depth comes from.** A planar tag's PnP depth is
essentially:

```
z ≈ fx · L / w_pixels
```

fx = focal length (from the intrinsic calibration), L = physical tag side
(`tag_size`), w = apparent tag width in pixels. Depth is therefore
proportional to the **fx·L product** — remember this for section 6.

**The fusion loop (default 20 Hz):**

1. **Freshness gate.** Each camera's latest pose is used only if younger
   than `max_age_sec`. Prevents pairing a live camera with a stale one.
2. **Transform to world.** Cam1's pose is already world (identity). Cam2's
   pose is multiplied through the extrinsic:
   `T_world_tag = T_world_cam2 ⊗ T_cam2_tag`.
   This is the extrinsic's entire job: making the two measurements
   comparable in one frame. Without it, averaging would mix numbers from
   two different coordinate systems.
3. **Discrepancy check.** `‖t1 − t2‖` between the two world-frame positions
   is published as `triangulation_error`. If it exceeds
   `max_discrepancy_m`, fusion falls back to cam1 only (a glitching camera
   should not be averaged in at half weight).
4. **Average.** Translation: linear mean. Orientation: hemisphere-corrected
   quaternion mean (quaternions q and −q encode the same rotation; one is
   flipped if their dot product is negative, then the normalised mean is
   taken — a good approximation of SLERP at 50/50 weight).
5. **Fallback.** If only one camera sees the tag, its pose is passed
   through unfused — the system degrades gracefully instead of dropping out.

**Character:** smooth, full 6-DoF, robust — but inherits any fx·L scale
error at first order.

---

## 5. Method B — Geometric triangulation, step by step

**Inputs:** `/camX/apriltag/detections` (for the tag's **centre pixel**
only) + `/camX/camera_info` (K, distortion) + TF (camera positions).

**The loop:**

1. **Centre pixel.** From each camera's detection message take
   `(u, v)` of the tag centre — the average of the four corners, which
   already halves pixel noise versus a single corner.
2. **Undistort.** `cv2.undistortPoints` removes lens distortion and returns
   normalised image coordinates `(x_n, y_n)`.
3. **Back-project a ray.** Direction in the camera frame:
   `d = normalize([x_n, y_n, 1])`. Note that intrinsics are used only to
   set this **direction** — no depth, no tag size involved.
4. **Move rays to world.** Each ray's origin is its camera's optical centre
   `C_i`; its direction is rotated by the camera's world orientation. Both
   come from the static TFs.
5. **Optional baseline anchor.** If `baseline_override > 0`, cam2's
   position is rescaled along the cam1→cam2 line so the baseline equals
   this tape-measured value. Legitimate here (unlike in Method A, where
   rescaling only the extrinsic breaks camera agreement) because ray
   intersection consumes no per-camera depths — this pins the output's
   scale to a physical tape measurement.
6. **Intersect.** Two rays in 3D almost never truly cross; the node finds
   the two closest points (one on each ray, via the common-perpendicular
   closed form) and publishes their **midpoint** as the position. The
   distance between those two points is published as `ray_gap`.
7. **Orientation.** Ray intersection is position-only, so the orientation
   field is copied from cam1's PnP pose.
8. **Degeneracy guard.** As the rays approach parallel (tag far away
   relative to the baseline) the intersection becomes ill-conditioned; the
   node warns and skips instead of publishing garbage.

**Character:** scale rides on the **baseline**, not fx·L; immune to
tag_size errors and to planar-pose-ambiguity flips (the centre pixel
doesn't move when the orientation solution flips). Trade-offs: noisier
per frame (pixel jitter → ray jitter), position-only, weak at long
range / small baseline.

---

## 6. Why two methods: the error model

| Error source | Method A (fusion) | Method B (geometric) |
|---|---|---|
| `tag_size` wrong | scales every distance, first order | **immune** (L unused) |
| `fx` wrong | scales every distance, first order | second order only (slightly bends rays; ~zero near image centre) |
| Pose-ambiguity flip (frontal planar tag) | corrupts pose | **immune** (centre pixel unmoved) |
| Corner pixel noise | amplified ~z/w per pixel into depth | enters as small angle; centre averaging helps |
| Extrinsic **rotation** error | hidden inside cam2's world pose | exposed, measurable as `ray_gap` |
| Extrinsic **baseline length** error | no effect on scale | first order into depth (hence `baseline_override`) |

The two methods are near-complementary. Consequences:

- `scale_check` (‖B‖/‖A‖) is a **live measurement of the fx·L error** — the
  system audits its own dominant error source continuously.
- `ray_gap` isolates the extrinsic's **angular** quality, which no Method-A
  topic can separate out.
- Recommended roles: **Method A as the primary tracking output** (smooth,
  6-DoF), **Method B as the referee** for scale and consistency.

---

## 7. Calibration workflow

Accuracy is bounded by three calibrations, in strict order.

### 7.1 Intrinsics (checkerboard) — per camera, separately

Determines K = [[fx,0,cx],[0,fy,cy],[0,0,1]] and distortion coefficients
for **each** physical camera. Same model ≠ same lens: never share one file
between two cameras.

```bash
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 \
    --ros-args -r image:=/image_raw -r camera:=/camera
```

Rules: cover the whole image incl. corners; include poses at your actual
working distances; 40+ accepted views; reprojection error < 0.5 px;
calibration resolution must equal the streaming resolution (640×480 here);
never touch zoom/focus afterwards.

### 7.2 Tag size (calipers)

`tag_size` is the side length of the **black square only** (outer edge of
the black border — not the printed sheet, not the white margin). Measure
with calipers. Depth scales linearly with this value, so a 4 mm error on a
14 cm tag is a 3% error on every distance. Print at 100% / "actual size".

### 7.3 Extrinsic (this package's script)

With the full launch running and the tag rigidly mounted where **both**
cameras see it:

```bash
ros2 run apriltag_triangulation measure_cam2_extrinsic_apriltag \
    --ros-args -p n_samples:=100 -p n_positions:=3 \
    -p known_baseline:=0.59 -p current_tag_size:=0.15
```

**Math.** Both cameras observe the same physical tag at the same instant,
so `T_cam1_tag = T_cam1_cam2 ⊗ T_cam2_tag`, giving
`T_cam1_cam2 = T_cam1_tag ⊗ inv(T_cam2_tag)` per sample. Samples are
averaged (translation linearly, rotations by hemisphere-corrected
quaternion mean) after MAD outlier rejection.

**Built-in safeguards** (each one exists because of a real failure we hit):

- *Freshness pairing* — a sample is taken only when both cameras' poses are
  fresh, so a camera that briefly lost the tag can't contribute stale data.
- *Spread gate* — on a static scene the baseline spread across samples
  should be millimetres; if it exceeds 2 cm the result is flagged
  untrustworthy (causes: something moved, shared TF frames, ambiguity
  flips). MAD alone can't catch a *smeared* distribution — it only removes
  outliers relative to the bulk.
- *Multi-position mode* (`n_positions>1`) — collects batches at several tag
  placements, auto-detecting when you move the tag and when it settles.
  Since the extrinsic is one fixed physical quantity, the per-position
  estimates must agree; their spread directly measures residual
  position-dependent error (intrinsics/distortion).
- *Scale diagnostic* — compares the measured baseline against your
  tape-measured `known_baseline`. It can only see the fx·L **product**, so
  it reports both interpretations: tag_size wrong (if unverified) or fx
  wrong (if tag_size is caliper-verified). It deliberately **never**
  rescales the result: the same scale error lives in every live detection
  too, so extrinsic and live data currently agree with each other —
  rescaling only the extrinsic would break that agreement at runtime.

**Recalibrate whenever anything changes**: camera moved/rotated, intrinsics
file changed, tag_size changed, resolution changed. The extrinsic describes
one specific measurement configuration.

---

## 8. Every ROS2 topic explained

### Final outputs

| Topic | Type | Meaning |
|---|---|---|
| `/apriltag/triangulated_pose` | PoseStamped | **Method A output.** Fused 6-DoF tag pose in `world`. Primary tracking output. |
| `/apriltag/geometric_pose` | PoseStamped | **Method B output.** Ray-intersection position in `world`; orientation copied from cam1's PnP. |

### Quality / diagnostic metrics

| Topic | Type | Meaning |
|---|---|---|
| `/apriltag/triangulation_error` | Float32 | Distance (m) between cam1's and cam2's world-frame position estimates this instant. Live health metric for the extrinsic + detections. Rising trend ⇒ something moved; recalibrate. |
| `/apriltag/triangulation_error_pct` | Float32 | Same, as % of the camera baseline (context: 3 cm means different things at 0.3 m vs 2 m baselines). Healthy: < ~5%. |
| `/apriltag/camera_baseline` | Float32 | Distance (m) between the two camera optical centres, from the static TFs. Published once at startup. |
| `/apriltag/ray_gap` | Float32 | Method B: miss distance (m) between the two rays at closest approach. Isolates extrinsic **rotation** error + pixel noise. Healthy: < 1–2 cm. |
| `/apriltag/scale_check` | Float32 | ‖geometric position‖ / ‖fused position‖. Live readout of the fx·L scale error: 1.00 = none; e.g. 0.945 ⇒ fusion over-ranges ~5.8%. Most meaningful with `geo_baseline_override` set to the tape-measured baseline. |

### Per-camera intermediate topics

| Topic | Type | Meaning |
|---|---|---|
| `/camX/apriltag_pose` | PoseStamped | Tag pose **in camera X's own frame** (raw PnP via TF, before any extrinsic). Inputs to Method A and to the calibration script. |
| `/camX/apriltag_detected` | Bool | True while the tag is visible to camera X with sufficient `decision_margin`. |
| `/camX/apriltag/detections` | AprilTagDetectionArray | Raw detector output: tag id, corner and centre pixels, homography, `decision_margin` (confidence). Method B reads its centre pixel from here. |
| `/apriltag/cam1_pose`, `/apriltag/cam2_pose` | PoseStamped | Each camera's estimate **transformed into world** by the fusion node — directly comparable to each other; their difference is `triangulation_error`. |

### Infrastructure

| Topic | Meaning |
|---|---|
| `/camX/image_raw` (+ `/compressed`, `/theora`, `/zstd`, …) | Camera stream; the suffixed ones are auto-generated `image_transport` variants — ignore. |
| `/camX/camera_info` | Intrinsics (K, distortion) loaded from the `ost.yaml` files. Consumed by apriltag_ros and Method B. |
| `/tf` | Dynamic transforms: `camX_optical_frame → tag0_camX` from each apriltag node, per detection. |
| `/tf_static` | The two static camera transforms from the launch arguments. |

---

## 9. Launch and parameters

```bash
ros2 launch apriltag_triangulation dual_apriltag_triangulation.launch.py \
    cam1_device:=/dev/video0 \
    cam2_device:=/dev/video2 \
    cam1_calib:=file:///home/ros/ws/src/camera_calibrations/camera_calib/ost.yaml \
    cam2_calib:=file:///home/ros/ws/src/camera_calibrations/camera_calib2/ost.yaml \
    tag_size:=0.15 \
    tag_id:=0 \
    cam2_tx:='-0.572726' cam2_ty:='0.073423' cam2_tz:='0.236441' \
    cam2_qx:='-0.074490' cam2_qy:='0.489397' cam2_qz:='0.010207' cam2_qw:='0.868814' \
    geo_baseline_override:=0.59
```

Startup is staggered on purpose: cam2 starts 3 s after cam1 (two identical
webcams grabbing the USB bus simultaneously fail with "Unable to start
stream"; `mjpeg2rgb` keeps per-camera bandwidth low), and the two
computation nodes start at 6 s.

| Argument | Meaning |
|---|---|
| `cam1_calib`, `cam2_calib` | Per-camera intrinsics files. **Must be different files for different physical cameras.** |
| `tag_size` | Caliper-measured black-square side (m). Passed as a parameter override so it genuinely reaches apriltag_ros (a bare tags.yaml value would silently win otherwise). |
| `cam2_tx…cam2_qw` | The extrinsic from the calibration script. |
| `geo_baseline_override` | Tape-measured camera distance (m) to anchor Method B's scale; 0 = use the extrinsic length as-is. |
| `cam1_depth_scale`, `cam2_depth_scale` | Optional per-camera range corrections (measured/true; 1.0 = off). Applied in the adapter so calibration and runtime stay consistent. Prefer fixing intrinsics over using these. |

Key node parameters (set in the launch file):

| Parameter | Node | Default | Notes |
|---|---|---|---|
| `max_discrepancy_m` | fusion | 0.15 | Camera-disagreement gate; set to ~10% of baseline. |
| `max_age_sec` | fusion / geometric | 0.5 / 0.2 | Must exceed the detection pipeline latency (measure with `ros2 topic delay /cam1/apriltag/detections`; ~0.7 s observed here ⇒ use ≥ 1.0). Parameters are read **once at startup** — `ros2 param set` on a running node has no effect. |
| `min_decision_margin` | adapter / geometric | 50 | Detection confidence gate (0–255). |
| `fusion_rate_hz` | fusion | 20 | Output rate. |

---

## 10. Known pitfalls (learned the hard way)

1. **Shared TF tag frame.** Both cameras publishing child frame `tag0`
   makes its parent flip-flop; lookups then route through `world` and the
   launch extrinsic leaks into "raw" measurements — calibration converges
   to whatever you launched with (a perfect self-reinforcing loop). Fix:
   unique frames `tag0_cam1` / `tag0_cam2` (already wired in).
2. **fx·L degeneracy.** Focal length and tag size enter depth only as a
   product. A constant distance-ratio error cannot be attributed to either
   one from fused data alone. Disambiguate per camera (single-camera
   ratio tests: tag_size errors are identical across cameras and setups;
   fx errors differ per camera and can even flip sign between geometries)
   or via `scale_check`.
3. **One calibration file for two cameras.** Silently wrong for at least
   one of them; caused double-digit scale errors here.
4. **Calibration resolution ≠ stream resolution.** fx/fy scale with
   resolution; a 1280×720 calibration used on a 640×480 stream is wrong by
   exactly that ratio.
5. **`tag_size` = black square only**, calipers not ruler; printers rescale
   unless told "actual size".
6. **Pipeline latency vs freshness gates.** ~0.7 s of camera+detector
   latency will silently starve any node whose `max_age_sec` is smaller —
   no errors, just empty topics. Measure with `ros2 topic delay`.
7. **Parameters cache at startup.** `ros2 param set` succeeds but changes
   nothing for values read once in `__init__`. Put them in the launch file.
8. **Never rescale only the extrinsic** to match a tape baseline (Method A
   path): the same scale error is in every live detection, so the pair is
   currently self-consistent; a one-sided fix breaks runtime agreement,
   worse with distance. (Method B's `baseline_override` is the exception —
   it consumes no per-camera depths.)
9. **Frontal planar tags** rock/flip between mirror PnP solutions. Tilt
   the tag 15–25° or converge the cameras (which also enforces obliquity
   everywhere). Method B is immune; Method A is not.
10. **Moving robot:** timestamp evaluation data with the message
    `header.stamp`, not arrival time — a 0.7 s latency at 0.5 m/s is 35 cm
    of phantom error otherwise. Force short exposure to limit motion blur.
11. **Copy-paste whitespace.** Multi-line commands pasted with non-breaking
    spaces produce arguments like `/dev/video0␣␣` — four different cryptic
    errors from one invisible cause. Retype or use a script.
12. **MAD can't reject a smeared distribution** — it only removes outliers
    relative to the bulk. That's what the spread gate is for: on a static
    scene, millimetres or it's wrong.