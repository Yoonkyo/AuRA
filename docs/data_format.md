# Data format

This page describes the data AuRA v1 writes: the robot-executed and human-baseline cases, the grasp files, the raw synchronized recordings, the run summaries and the calibration results that the pipeline reads. The calibration tools' other outputs (overlay images, the hand-eye debug files, the re-anchoring results and extrinsics backups, the fingertip fit) are described in [calibration.md](calibration.md). The post-processing stage of v2 ([coming soon](../aura/postprocessing/README.md)) reads the robot-executed cases.

## Where data goes

| Data | Location | Written by |
|---|---|---|
| Robot-executed case | `$AURA_DATA_ROOT/franka_lift/<case>/` | `scripts/anygrasp_poses_one_shot.sh`, then `scripts/run_anygrasp_to_lift_one_shot.sh` |
| Human-baseline case | `$AURA_DATA_ROOT/hand_collected/<case>/` | `scripts/collect_hand_phystwin_one_shot.sh` |
| Calibration captures, teleoperation sessions | the `--output-root` you pass ([calibration.md](calibration.md) uses `$AURA_DATA_ROOT/calibration_captures/`) | `aura/capture/record_3oak.py`, `aura/robot/gamepad_record_franka_3oak.py` |
| Hand-run diagnostics | the output path you pass (the examples in [troubleshooting.md](troubleshooting.md) and in the module docstrings use `$AURA_DATA_ROOT/debug/`); pass the filter both `--out` and `--report`, or it overwrites the case's safe list and report | `aura/perception/aura_segment_static_target_image.py`, `aura/capture/anygrasp_data_collection.py`, `aura/grasping/filter_grasps_table_collision_pyroki.py` |

- **Data root.** `AURA_DATA_ROOT` defaults to `data/` in the repository, which is git-ignored. `--data-root` replaces the `franka_lift` or `hand_collected` folder for one run; pass the same value to the preparation script and the lift one-shot of a case.
- **Case names** must match `[A-Za-z0-9_-]+`. For robot-executed cases the segmentation prompt is derived from the name: a trailing `_<digits>` is removed, underscores become spaces, and a period is appended. `green_toy_1` becomes `"green toy."`, so name cases `<description>_<index>`. `--sam-prompt` overrides the prompt.

## Conventions

| Quantity | Convention |
|---|---|
| Lengths | metres, except depth maps and keys ending in `_mm` (millimetres) |
| Depth maps | **`uint16` millimetres**, 0 where there is no depth. Divide by 1000.0. Nothing in `metadata.json` states the unit. |
| World frame | the ChArUco board frame of the extrinsic calibration: table plane at z = 0, **+z pointing down** into the table, so points above the table have negative z ([details](calibration.md#the-world-frame)) |
| Robot base frame | FR3 base frame, +z up. `T_world_from_base` is in `robot_calib_result.npz`. |
| Camera frame | OpenCV pinhole: x right, y down, z along the optical axis |
| Camera index | the order of `AURA_OAK_IDS`; cameras are named `oak_<MXID>` |
| Images | undistorted on the camera; do not undistort them again. Use `K` alone (one per camera and resolution), as the point-cloud fusion does with the `intrinsics` of `metadata.json`. The residual `dist` stored with the intrinsics is used only by the calibration tools, in their PnP solves, reprojection errors and overlays ([calibration.md](calibration.md#2-camera-extrinsics)). |
| Roll, pitch, yaw | radians, `R = Rz(yaw) Ry(pitch) Rx(roll)` |
| Timestamps | seconds of the host's monotonic clock, comparable only within one recording. Wall-clock times are ISO strings: the `*_wallclock` keys, `created_at` in `anygrasp/anygrasp_collection_summary.json`, and `created_at` in `phystwin/fr3_home_state.npz` (when that start state was measured, not when the case was recorded). |

**Grasp poses.** AnyGrasp poses are in the frame of the input point cloud, which is the world frame. Column 0 of `rotation_matrix` is the approach axis, column 1 the finger-closing axis, and `translation` is the grasp centre. The executed tool-centre-point (TCP) pose is `R_tcp = R_anygrasp @ Ry(+90°)`. The preparation script sets this with `--rot-offset-rpy 0 1.57079632679 0 --offset-mode post` for the filter, and the executor uses the same `--rot-offset-rpy`. As a result, the TCP z axis is AnyGrasp's approach axis, the TCP y axis its closing axis, and the TCP x axis is minus AnyGrasp's z axis. The robot moves the TCP to the grasp centre itself, plus `--final-z-offset`; AnyGrasp's `depth` is not applied. The hover, final and lift offsets (`--hover-z-offset` 0.10, `--final-z-offset` 0, `--lift-z-offset` 0.15) are added along the **base** z axis (up), while `--table-z` and `--clearance-m` are measured along the **world** z axis (down).

## Robot-executed case

```
$AURA_DATA_ROOT/franka_lift/<case>/
├── anygrasp/                         static pre-interaction capture (stages 01-02), PhysTwin format, one frame
│   ├── color/{0,1,2}/0.png  depth/{0,1,2}/0.npy  source_meta/0.json
│   ├── calibrate.pkl  metadata.json  conversion_metadata.json  anygrasp_collection_summary.json
│   ├── mask/                         segmentation, see below
│   ├── shape/masked_image.png        copy of camera 0's RGBA mask; not read by v1
│   └── observation/                  fused object cloud: observation_frame000_filtered.{npz,ply}, ...
├── anygrasp_results/                 AnyGrasp output (stage 03) and filter output (stage 04)
├── anygrasp_results_no_collision/    only if stage 03 found no grasp and the retry 03b ran
├── logs/                             01_collect_anygrasp_frame.log ... 04_pyroki_filter.log
├── _work/anygrasp_collection/        raw_recording/ (the snapshot's bundles) and phystwin_3oak/ (intermediate)
├── one_shot_summary.json             preparation summary
├── phystwin/                         the grasp-and-lift recording, PhysTwin format
│   ├── color/{0,1,2}/<frame>.png  color/{0,1,2}.mp4  depth/{0,1,2}/<frame>.npy  source_meta/<frame>.json
│   ├── calibrate.pkl  metadata.json  conversion_metadata.json
│   ├── camera_calibration/{intrinsic,extrinsic}/oak_<MXID>.{npz,json}
│   ├── robot_calib_result.npz  fr3_home_state.npz  franka_fingertip_table_calibration_simple.json
│   ├── raw_recording/                synchronized bundles of the lift, see Raw recordings
│   ├── recording_summary.json  session_summary.json      copies from raw_recording/
│   └── README.txt                    written by the preparation script
└── global_one_shot_summary.json      execution and recording summary
```

- **Stage logs.** The preparation script writes one log per stage. The optional retry logs to `03b_anygrasp_pose_generation_no_collision.log`. The lift one-shot writes no log files of its own.
- **Preserved inputs.** Do not re-run the lift one-shot on a recorded case after a recalibration. At the end of each run, the lift one-shot copies the calibration, start-state and fingertip files it was given into `phystwin/`, and stops with `Refusing to overwrite different preserved file` if a different copy is already there. By then, a re-run after a recalibration has run the filter again (default `--refilter`), which rewrites the safe list the recorded grasp was taken from (`anygrasp_results/grasps_table_ik_safe.json`) and its report, and has rebuilt `calibrate.pkl` and `metadata.json` from the new calibration. [troubleshooting.md](troubleshooting.md#execution-and-lift-recording) shows how to rebuild `calibrate.pkl` and `metadata.json` from the case's own copies; the rewritten safe list and report may then no longer match the recording. If an earlier run stopped before making the copies, the next run copies the current files without a check. The recording executor also keeps the robot calibration it used in `raw_recording/robot_calib_result.npz`.
- **Retry output.** `anygrasp_results_no_collision/` holds only AnyGrasp's own files. The filter output (`grasps_table_ik_safe.json`, `grasps_table_ik_report.json`) always goes to `anygrasp_results/`.

## Human-baseline case

```
$AURA_DATA_ROOT/hand_collected/<case>/
├── color/{0,1,2}/<frame>.png  color/{0,1,2}.mp4  depth/{0,1,2}/<frame>.npy  source_meta/<frame>.json
├── calibrate.pkl  metadata.json  conversion_metadata.json
├── camera_calibration/{intrinsic,extrinsic}/oak_<MXID>.{json,npz}
├── raw_recording/                    bundles from aura/capture/record_3oak.py + recording_summary.json
├── raw_preview_3oak_numbered.mp4     the three views side by side with frame numbers, for choosing the range
├── recording_console.log             the recorder's console output
└── hand_collection_summary.json
```

The case contains the frames you selected (both ends inclusive), renumbered from 0. Human-baseline cases carry no robot state.

## PhysTwin capture format

The `anygrasp/` and `phystwin/` folders of a robot case and the human-baseline case folder share this layout:

| Path | Content |
|---|---|
| `color/<cam>/<frame>.png` | 8-bit, 3-channel image, 720 × 1280 at `high`. OpenCV reads it in BGR order. It is a lossless re-encode of the JPEG in the raw bundle. |
| `color/<cam>.mp4` | preview video at the recording frame rate (lift and human-baseline cases) |
| `depth/<cam>/<frame>.npy` | `uint16` millimetres, H × W, pixel-aligned with the colour image |
| `source_meta/<frame>.json` | a copy of the raw bundle's `meta.json`, including the robot state of robot recordings ([Raw recordings](#raw-recordings)) |
| `calibrate.pkl` | pickled Python list, in camera-index order, of 4 × 4 camera-to-world matrices stored as nested lists of floats. `np.asarray` gives shape (3, 4, 4). These are the `phys_twin_c2w` matrices of the extrinsic calibration. |
| `metadata.json` | camera metadata, see below |
| `conversion_metadata.json` | the conversion record, see below |
| `camera_calibration/` | copies of the intrinsic and extrinsic files used (`phystwin/` and human-baseline cases) |
| `mask/` | segmentation of the static capture (`anygrasp/` only) |

`<frame>` is the position in the converted range and always starts at 0. `conversion_metadata.json` (`frames[i].bundle_name`) and `source_meta/<frame>.json` (`bundle_index`) link each frame to its raw bundle.

### Camera metadata (`metadata.json`)

Written by `aura/conversion/calibration_matrix_to_phystwin_style.py`:

| Key | Content |
|---|---|
| `camera_names` | `oak_<MXID>` for each camera index |
| `serial_numbers` | the same strings |
| `frame_num` | number of frames |
| `intrinsics` | one 3 × 3 `K` per camera |
| `WH` | `[width, height]`, shared by all cameras |
| `per_camera` | for each camera: `camera_name`, `intrinsic_source` and `extrinsic_source` (file paths), `input_WH`, `output_WH`, `crop`, `dist` (residual distortion, not applied), `z_positive_fix_applied` (false), `z_positive_fix_axis` (null), `phys_twin_c2w_original`, `phys_twin_c2w_written` |

Read keys by name instead of assuming exactly this set of keys.

### Conversion record (`conversion_metadata.json`)

`aura/conversion/raw_data_to_phystwin_style.py` writes its own `metadata.json`, and the calibration converter, which runs next, replaces it. The pipeline scripts copy the first file to `conversion_metadata.json` in between; do the same if you run the two converters by hand. Its keys are:

- `dataset_name`: the name of the folder the converter wrote into, which is `phystwin` in `phystwin/`, `phystwin_3oak` in `anygrasp/` (converted in `_work/anygrasp_collection/phystwin_3oak/` and then copied), and the case name in human-baseline cases;
- `source_recording`, `camera_order`, `camera_index_map`, `num_cameras`, `num_frames`;
- `frame_selection`: `start_frame`, `end_frame` (exclusive; `null` in `phystwin/`, where the whole recording is converted), `stride` and `selected_bundle_names`;
- `rgb_filename`, `depth_filename`: the input file names in each camera folder of a raw bundle (`rgb.jpg`, `depth.npy`);
- `input_image_sizes`, `output_image_sizes`, `crop_by_camera`;
- `generated_layout`: the output path templates `color/{camera_index}/{frame_index}.png` and `depth/{camera_index}/{frame_index}.npy`;
- `frames`: per frame, the source bundle and the source and destination file paths.

In `anygrasp/`, `source_recording` and the paths in `frames` point into `_work/anygrasp_collection/`, not into `anygrasp/`.

## Segmentation and observation cloud

Stage 02 writes these files into `anygrasp/`. The observation cloud is the only input of AnyGrasp.

| Path | Content |
|---|---|
| `mask/cam<N>_frame0_sam_rgba.png` | SAM 2.1 mask of camera `<N>`: RGB inside the mask, alpha = mask |
| `mask/cam<N>_frame0_sam_rgba.detections.jpg` | the GroundingDINO boxes and the selected one |
| `mask/cam<N>_frame0_sam_rgba.selection.json` | prompt, thresholds, box counts before and after NMS, the selected box, the SAM score and the mask size |
| `mask/<N>/<component>/0.png` | one `uint8` mask (0 or 255) per 8-connected component of at least 20 px; component 0 is the largest |
| `mask/mask_info_<N>.json` | `category`, `union_pixels`, `num_components`, `components` (pixels and rank of each), `selected_for_fusion` |
| `mask/mask_info_all_cameras.json` | the per-camera files combined |
| `observation/observation_frame000_filtered.npz` | the fused, filtered object cloud (below) |
| `observation/observation_frame000_filtered.ply` | the same points and colours |
| `observation/raw_to_fused_observation_run_meta.json` | `category` (the prompt), `mask_object_id`, and `fusion_meta`: `raw_num_points`, `filtered_num_points`, `per_camera_stats`, and `filter_meta` with the parameters of each filter and the point counts before and after it |
| `observation/visualization/` | `final_overlay_cam<N>.png`, `topdown_final.png`, `topdown_filter_stages.png`, `observation_visualization_summary.json` |

`observation_frame000_filtered.npz` has three keys:

| Key | Shape, dtype | Content |
|---|---|---|
| `points` | (N, 3) `float32` | metres, world frame (+z down, table at z = 0) |
| `colors` | (N, 3) `float32` | RGB in [0, 1] |
| `observation_cams` | (3,) `int32` | the fused cameras |

The cloud is built from the masked pixels with depth between 0.2 and 1.5 m. It is then filtered in this order: table cut (keep z ≤ `--table-z` + `--table-eps`, i.e. at most 1 cm below the table), 3 mm voxel grid, statistical outlier removal (35 neighbours, 1.2 σ), radius outlier removal (10 neighbours within 6 mm), and a crop to the 1st to 99th percentile along the cloud's principal axes, with no margin (`--pca-margin-long 0 --pca-margin-cross 0`). These are the values the preparation script passes (`--help` lists them); the defaults of `aura/perception/phystwin_data_to_observation_pointcloud.py` differ. On the authors' `colorful_rope_6` case, 29,981 masked points were reduced to 4,539.

## AnyGrasp output

`anygrasp_results/grasps.json` is a JSON list, ordered by score after non-maximum suppression and cut to `--top-k` (20) entries:

```jsonc
{
  "rank": 0,                              // position in this list; 0 = highest score
  "score": 0.1388,
  "width": 0.08,                          // gripper opening (m), at most --max-gripper-width (0.08)
  "height": 0.03,                         // gripper height (m)
  "depth": 0.01,                          // grasp depth along the approach axis (m)
  "translation": [-0.136, 0.220, -0.023], // grasp centre, metres, world frame (+z down)
  "rotation_matrix": [[...], [...], [...]], // column 0 = approach axis, column 1 = closing axis
  "pose_matrix_4x4": [[...], [...], [...], [...]],
  "object_id": -1
}
```

| File | Content |
|---|---|
| `best_grasp.json` | entry 0 of `grasps.json`. This is not necessarily the grasp that gets executed. |
| `result_summary.json` | `input`, `num_input_points`, `num_grasps_after_nms` (despite the name, the number of entries in `grasps.json`, so at most `--top-k`), `lims` (workspace box = cloud bounds ± 0.02 m), `best_score`, `best_translation`, `best_rotation_matrix` |
| `grasps_raw.npy` | (N, 17) `float64`, one row per grasp: score, width, height, depth, rotation (9 values, row-major), translation (3), object id |
| `grasp_scores.npy`, `grasp_widths.npy`, `grasp_heights.npy`, `grasp_depths.npy`, `grasp_rotations.npy`, `grasp_translations.npy`, `grasp_pose_matrices.npy`, `grasp_object_ids.npy` | the same values, one field per file |
| `debug_all_grasps.png`, `debug_best_grasp.png` | renders of the candidates on the cloud, flipped in z as in the AnyGrasp demo. Only the pictures are flipped. |

If AnyGrasp finds no grasp, only `result_summary.json` is written, with `{"num_grasps": 0, "lims": [...]}`. The preparation script then retries without collision detection and writes the retry's files to `anygrasp_results_no_collision/`.

## Feasibility filter output

### Safe list (`grasps_table_ik_safe.json`)

This file lists the accepted grasps, and it is what the executor reads. Each entry is the original `grasps.json` entry plus a `table_ik_fk_check` block.

- **Order.** Entries are sorted by the joint motion from home to hover (`home_to_hover_cartesian_robustness.total_joint_motion_l1_rad`, ascending; ties by descending score). This is the default `--safe-order joint_motion` of the filter.
- **Indices.** The position in this list is what the lift one-shot calls `--safe-index` and the executors `--rank`. The entry's own `rank` field is the original AnyGrasp rank.

| `table_ik_fk_check` field | Content |
|---|---|
| `index`, `rank`, `score` | position in `grasps.json`, AnyGrasp rank and score |
| `accepted`, `rejection_stage`, `reason` | `true`, `null`, `null` in the safe list |
| `coarse_grasp_center_check` | the TCP target lies above the table (world z < `--table-z`) |
| `target_fingertip_table_check` | `worst_tip`, `minimum_clearance_m`, `required_clearance_m`, `tip_positions_world_m` and `tip_clearances_m` of the two fingertips at the grasp |
| `nominal_target_T_world_from_tcp`, `target_T_world_from_tcp` | TCP pose at the grasp in the world frame, before and after `--final-z-offset` |
| `target_camera_keepout_check` | distances of five tool points (the TCP, both fingertips, and the points halfway between the TCP and each fingertip) to each camera centre, against `--camera-keepout-radius-m` |
| `home_to_hover_cartesian_robustness` | dense IK along the straight Cartesian path from home to hover. Summary values: `safe`, `total_joint_motion_l1_rad`, `endpoint_max_joint_delta_rad`, `max_waypoint_joint_delta_rad`, `cartesian_rotation_deg`, `minimum_soft_joint_margin_rad`, `minimum_tip_table_clearance_m`, `minimum_camera_distance_m` and the thresholds used. Per waypoint (`waypoints`): joint angles, target and forward-kinematics poses, errors, joint-limit margins, fingertip and camera checks. |
| `ik_fk_endpoints` | `hover`, `final` and `lift`: the IK solution (`q_arm_rad`), `target_T_base_from_tcp`, `fk_T_base_from_tcp`, position and orientation errors, joint-limit checks |
| `cartesian_local_path_robustness` | the segments `hover_to_final` and `final_to_lift`, 25 samples each (`--path-samples`) |

The joint angles in this block only validate the path; the executor moves with Cartesian control and never sends them. Before it moves, the recording executor (`aura/robot/execute_anygrasp_franka_record_lift.py`) checks that `home_to_hover_cartesian_robustness.safe` is `true` and that `ik_fk_endpoints.hover.target_T_base_from_tcp` agrees within 1 mm and 0.5° with the hover pose it computes itself. It therefore rejects entries of `grasps.json`, as well as a safe list made with a different `--hover-z-offset`, `--rot-offset-rpy` or `robot_calib_result.npz`. It does not compare the final and lift poses: a safe list made with a different `--final-z-offset` or `--lift-z-offset` passes, and the robot then descends or lifts to a pose that the filter never checked. Run the filter and the executor with the same offsets ([troubleshooting.md](troubleshooting.md#execution-and-lift-recording)).

### Report (`grasps_table_ik_report.json`)

| Key | Content |
|---|---|
| `summary` | `input_grasps`, `accepted_grasps`, `rejected_grasps`, `accepted_ranks`, `accepted_output_ranks` (safe-list order), `safe_order`, `rejection_counts`, `no_robot_motion_commanded` |
| `configuration` | the options of the run (for `--calib-results`, the robot and fingertip calibration files it resolved to), the camera keep-outs, the soft joint limits and `path_check_scope`. Not recorded: the IK weights `--pos-weight`, `--ori-weight` and `--rest-weight`, `--open-finger-q` (the finger value used is the last entry of `current_state.q_all_actuated_rad`), `--include-tracebacks` and the output paths |
| `current_state` | the start configuration from the state file and its joint-limit checks |
| `entries` | one per input grasp, with the fields above up to the first failed check |

The checks run in this order, and only the first failure of a grasp is recorded: `coarse_world_z`, `target_fingertip_table`, `target_camera_keepout`, `home_to_hover_cartesian_robustness`, `ik_fk_hover`, `cartesian_hover_to_final_robustness`, `cartesian_final_to_lift_robustness`. An error while checking a grasp is recorded as `exception`; `--include-tracebacks` adds the traceback.

## Raw recordings

All three recorders write the same bundle layout:

```
<output-root>/
├── bundle_000000/
│   ├── meta.json
│   └── oak_<MXID>/                   one folder per camera
│       ├── rgb.jpg
│       └── depth.npy
├── bundle_000001/ ...
├── recording_summary.json
├── session_summary.json              recording executor only: identical copy of recording_summary.json
├── robot_calib_result.npz            recording executor only: byte copy of --calib
└── robot_states_raw.npz              teleoperation recorder only
```

| Recorder | Typical output root |
|---|---|
| `aura/capture/record_3oak.py` (cameras only) | calibration captures, `_work/anygrasp_collection/raw_recording/` of the static snapshot, `raw_recording/` of a human-baseline case |
| `aura/robot/execute_anygrasp_franka_record_lift.py` | `phystwin/raw_recording/` of a robot-executed case |
| `aura/robot/gamepad_record_franka_3oak.py` | the teleoperation session's `--output-root` |

- **Images.** `rgb.jpg` is a JPEG, quality 90 (camera-only and teleoperation recorders) or 85 (recording executor). This is the only lossy step of the pipeline. `depth.npy` is `uint16` millimetres, aligned with the RGB image.
- **Numbering.** Bundles are numbered consecutively from 0 in time order. Timesteps of the 15 Hz grid that were dropped during synchronization are skipped, not left as gaps, so consecutive bundles can be more than one period apart. `target_ts` (robot recorders) and `sources.<camera>.device_ts` (capture times) give the real times.
- **Lift recordings** cover only the lift: they start when the measured TCP has moved 1.5 mm (`--motion-start-delta-m`) and end 1 s after the lift (`--record-after-lift`). That is about 3 s, or about 40 grid timesteps at 15 fps.

### Bundle metadata (`meta.json`)

| Field | Recorders | Content |
|---|---|---|
| `bundle_index` | all | the bundle number |
| `target_ts` | robot recorders | grid time of this timestep |
| `target_ts_estimate` | camera-only | estimated grid time. It is computed from the bundle number, so it is too early after the first dropped timestep; use `sources.<camera>.device_ts` (the capture time) instead. |
| `match_span_ms` | all | spread of the cameras' `match_ts`. In the camera-only recorder this is not the capture-time spread, which can be much larger ([Known limitations](troubleshooting.md#recording-and-cameras)) |
| `host_span_ms` | all | spread of the times at which the frames reached the host |
| `sources.<oak_MXID>` | all | `match_ts` (the time used for matching: `device_ts` in the robot recorders; in the camera-only recorder `device_ts` plus that camera's receive latency, frozen at the end of the warm-up as `projectors.<camera>.fixed_offset`), `host_ts`, `device_ts` (the capture time, on the host clock despite the name), `frame_number`, `rgb_shape`, `depth_shape`, `depth_dtype`, offsets from the earliest camera; `target_error_ms` in the robot recorders |
| `robot` | robot recorders | robot state at this timestep (below) |
| recording parameters, `projectors`, `offline_sync` (`true`) | all | diagnostics. The robot recorders also store `offline_sync_meta_preview` (`sync_fps`, `sync_tolerance_ms`, `period_s`, `chosen_phase_s`); the camera-only recorder and the recording executor also store `per_source_match_offsets_ms` (each camera's `match_ts` offset from the earliest, in ms). The teleoperation recorder also stores `robot_ip`. |

The robot recorders sample the robot state at 100 Hz (`--state-publish-hz` of the recording executor, `--robot-state-hz` of the teleoperation recorder) from the robot's 1 kHz state stream, and interpolate it to each bundle:

| `robot` field | Recording executor | Teleoperation recorder |
|---|---|---|
| `joint_angles_rad` | 7 joint angles (rad), linearly interpolated | the same |
| `pose_xyzrpy` | end-effector pose (`O_T_EE`) in the base frame: x, y, z (m), roll, pitch, yaw (rad). Positions interpolated linearly, angles along the shortest path. | not recorded |
| `sync_target_ts` | the time the state is interpolated to: the bundle's `target_ts` | the median of the cameras' `match_ts` |
| `host_ts` | equal to `sync_target_ts` when interpolated; the sample's own time when the nearest sample is used | the same |
| `sync_error_abs_ms` | distance between `sync_target_ts` and the nearest raw sample: at most about 5 ms when interpolated at 100 Hz, possibly much more when the nearest sample is used | the same |
| `interpolation_extrapolated` | `true` when the bundle lies outside the sampled robot states; the nearest sample is then used. This is common for the first few bundles, whose frames can be captured up to a few tenths of a second before robot sampling starts: check this field or `sync_error_abs_ms` before you use them. | the same |
| `state_source` / `state_type` | `state_source`: `fci_state_interpolated_to_camera_capture` when interpolated, `actual_measured_shared_state` when the nearest sample is used | `state_type`: `joint_angles_interpolated` or `joint_angles_nearest_endpoint` |
| `control_command_success_rate` | not recorded | success rate of the 1 kHz control loop |
| `interpolation_alpha`, `interpolation_left_index`, ... | interpolation details | the same |

### Recording summary (`recording_summary.json`)

- **Every recorder** writes `created_at_wallclock`, `record_start_wallclock`, `depthai_version`, `args` (all recorder arguments), `raw_frame_counts` (frames per camera), `reader_errors`, `projectors` and `offline_sync`.
- **`offline_sync`** holds `sync_fps`, `sync_tolerance_ms`, `period_s`, `chosen_phase_s`, `num_target_timesteps`, `num_saved_timesteps`, `num_dropped_timesteps`, `dropped_targets_preview` (reasons), the `match_span_ms_*` percentiles and the `target_error_ms_abs_*` percentiles.
- **Robot recorders** also write `robot_states` (count, time range and gap percentiles of the robot samples) and `save_summary` (saved bundles and `robot_sync_error_abs_ms_*` percentiles).
- **The recording executor** also writes `calibration_manifest` (one entry per `oak_<MXID>.npz` in `$AURA_CALIB_ROOT/intrinsic_results/<WxH>/`, also for cameras that were not recorded: `K`, `dist`, image size and the file read; `{}` if it reads none), `robot_calibration_copy` (source path, size and the transforms used) and `stopped_at_wallclock`. Its `session_summary.json` is an identical copy.

`robot_states_raw.npz` (teleoperation recorder) holds every robot sample of the recording: `host_ts` (N,), `joint_angles_rad` (N, 7) and `control_command_success_rate` (N,), all `float64`.

## Run summaries

| File | Written by | Content |
|---|---|---|
| `<case>/one_shot_summary.json` | preparation script | prompt, input and output paths, `num_selected_anygrasp_poses`, `safe_grasp_count`, the state file and calibration passed, `static_sync_tolerance_ms`, `observation_filter` (the voxel, outlier, PCA and table-filter values and the cameras; not `--mask-object-id`), `"robot_motion_commanded": false` |
| `<case>/global_one_shot_summary.json` | lift one-shot, at the end of a complete run | `safe_index`, `safe_grasps`, `home_enabled`, `home_after_enabled`, `release_after_enabled`, `auto_confirmed`; `dynamic_recording` with `num_frames`, `resolution`, frame rates, `sync_tolerance_ms`, `hover_speed`, `descend_speed`, `lift_speed` (m/s), `cartesian_runtime_joint_margin_rad`, `refilter_enabled`, `motion_start_delta_m`, `record_after_lift_s`; `outputs` (paths) |
| `<case>/anygrasp/anygrasp_collection_summary.json` | static capture (stage 01) | resolution, frame rates, `record_duration`, `capture_bundle_index`, `oak_ids`, `camera_order`, calibration folders, `"robot_calib_used": false`, the commands run |
| `<case>/hand_collection_summary.json` | human-baseline script | `selected_start_frame_inclusive`, `selected_end_frame_inclusive`, `selected_frame_count`, `resolution`, `fps`, `sync_tolerance_ms`, `offline_sync` of the recording, paths |

Both one-shot summaries hold the options of the last call that wrote them, not necessarily those the case's data were made with:

- **`one_shot_summary.json`** is rewritten by every call, also by a resume that skips stages 01-03 (every lift run on a prepared case), so its prompt and its capture, fusion, state-file and calibration values are those of the last call. The values the outputs were made with are in `anygrasp/anygrasp_collection_summary.json` (the capture values; the sync tolerance is in `commands`), `anygrasp/observation/raw_to_fused_observation_run_meta.json` (the prompt as `category`, `mask_object_id` and the fusion values) and the `configuration` and `current_state` of `anygrasp_results/grasps_table_ik_report.json` (the filter's values, state file and calibration).
- **`global_one_shot_summary.json`** is rewritten by every complete run, also by one that reuses an existing raw recording, so its options can differ from the recording's. The recording's own values are under `args` in `phystwin/recording_summary.json` (`rank` = `--safe-index`, `motion_speed`, `descend_speed`, `lift_speed`, `record_*`); the release, home-after and refilter settings are recorded nowhere else.

## Calibration files

The procedure that writes these files is in [calibration.md](calibration.md).

| File | Keys |
|---|---|
| `intrinsic_results/<WxH>/oak_<MXID>.npz` | `K` (3, 3), `dist` (1, 5), `rms` (1,), `image_size` `[width, height]` |
| `intrinsic_results/<WxH>/oak_<MXID>.json` | `K`, `dist`, `image_width`, `image_height`, `rms_reprojection_error`, accepted views, `filter_stats`, `board`, `thresholds`, `per_view_errors` |
| `extrinsic_results/oak_<MXID>.{npz,json}` | `K`, `dist`, `rvec_cam_from_board`, `tvec_cam_from_board`, `T_cam_from_board`, `T_board_from_cam`, `T_cam_from_world`, `T_world_from_cam`, `phys_twin_c2w` (= `T_world_from_cam`), `image_size`, `num_markers`, `num_charuco`, `reproj_rms` |
| `robot_calibration_results/robot_calib_result.npz` | 4 × 4 `float64`: `T_world_from_camera`, `T_world_from_base`, `T_base_from_world`, `T_ee_from_board`, `T_board_from_ee` |
| `robot_calibration_results/robot_calib_result.json` | the same matrices, plus `counts`, `filters`, `error_summary`, `per_sample_errors` and the scan logs |
| `.../fingertip_calibration_results/franka_fingertip_table_calibration_simple.json` | `trajectories` (the table contacts) and `end_effector_fingertip_translation` (`right_outer_tip_in_ee_m`, `left_outer_tip_in_ee_m`, in metres and as `*_mm`) |
| `assets/fr3_home_state.npz` | `q` (7,): the start configuration of the grasp filter; see [`assets/README.md`](../assets/README.md) |

## Reading a case in Python

Run this in the `AuRA` env, after `set -a; source config/aura.env; set +a`:

```python
import json, os, pickle
from pathlib import Path

import cv2
import numpy as np

case = Path(os.environ.get("AURA_DATA_ROOT", "data")) / "franka_lift" / "<case>" / "phystwin"
meta = json.loads((case / "metadata.json").read_text())
c2w = np.asarray(pickle.load(open(case / "calibrate.pkl", "rb")))          # (3, 4, 4)

cam, frame = 0, 0
bgr = cv2.imread(str(case / "color" / str(cam) / f"{frame}.png"))           # (H, W, 3) uint8, BGR
depth = np.load(case / "depth" / str(cam) / f"{frame}.npy") / 1000.0      # (H, W) metres
K = np.asarray(meta["intrinsics"][cam])

v, u = np.nonzero(depth > 0)
z = depth[v, u]
pts_cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
pts_world = pts_cam @ c2w[cam][:3, :3].T + c2w[cam][:3, 3]                 # world frame, +z down
colors = bgr[v, u, ::-1] / 255.0                                            # RGB in [0, 1]

robot = json.loads((case / "source_meta" / f"{frame}.json").read_text())["robot"]
q = np.asarray(robot["joint_angles_rad"])                                   # joint angles at this frame
```

## Before sharing data

Several files record details of the machine they were made on. Scrub or regenerate them before you publish data:

- **Absolute paths.** `conversion_metadata.json`, `metadata.json` (`per_camera.*_source`), `mask_info_*.json`, `*.selection.json`, `raw_to_fused_observation_run_meta.json`, `result_summary.json`, `grasps_table_ik_report.json` (`configuration`), every `*summary.json`, `meta.json` of lift recordings and its copies `phystwin/source_meta/<frame>.json` (`robot_calibration_copy`), the calibration JSON reports, including `robot_calib_result.json`, and the console logs.
- **Robot address.** `meta.json` of teleoperation recordings (`robot_ip`; also in its copies `source_meta/<frame>.json` if you convert such a recording), the `args` of `recording_summary.json` / `session_summary.json` of robot recordings, and a lift one-shot log kept with `tee`.
- **Camera serial numbers (MXIDs).** Folder and file names (`oak_<MXID>`), `metadata.json` (`camera_names`, `serial_numbers`, `per_camera`), `conversion_metadata.json` (`camera_order`, `camera_index_map`, per-camera keys and file paths), every `meta.json` and its copies `source_meta/<frame>.json`, `recording_summary.json` / `session_summary.json` (`args`, per-camera keys such as `raw_frame_counts` and `reader_errors`, `offline_sync.dropped_targets_preview`), `anygrasp_collection_summary.json` (`oak_ids`, `camera_order`, `serial_numbers`, `commands`), `hand_collection_summary.json` (`offline_sync.dropped_targets_preview`), `grasps_table_ik_safe.json` and `grasps_table_ik_report.json` (camera keep-out names, `closest_camera`, `distances_m`), the calibration JSON reports (`camera_name`, image paths) and the console logs. The MXIDs appear bare or as `oak_<MXID>`, as keys and as values, so search the whole case for each MXID rather than relying on this list.
- **Console logs.** `logs/*.log` of robot cases (one per preparation stage), `recording_console.log` of human-baseline cases and any lift one-shot log you kept with `tee` (the command in [usage.md](usage.md#step-2-full-autonomous-run) writes it to `logs/<case>.log`, outside the case folder) are verbatim console output. The lift one-shot runs the preparation script first, so its log also contains the output of the preparation stages that ran. Leave these logs out of published data, or scrub them. They contain:
  - absolute paths, which usually include your user name: the repository, data, calibration and AnyGrasp SDK folders, and the Python environments;
  - the MXIDs, in `01_collect_anygrasp_frame.log`, `recording_console.log` and a lift one-shot log (the recording executor prints them when it opens the cameras);
  - the AnyGrasp license check messages in `03_anygrasp_pose_generation.log` and `03b_anygrasp_pose_generation_no_collision.log`, which give the path of your license folder and the names of your `<name>.lic`, `<name>.public_key` and `<name>.signature` files;
  - the robot address, only in a lift one-shot log: stage `02b_release_and_return_home` prints `[robot] connecting to <ip>`.
