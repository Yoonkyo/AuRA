# Calibration

Every rig needs its own calibration. Do the steps in this order: camera intrinsics, then extrinsics, then hand-eye. The fingertip geometry is independent of the other three, and re-anchoring is only needed after a camera has moved. The hardware, including the ChArUco board, is described in [hardware.md](hardware.md). All commands run from the repository root.

## What you produce

All results go to `$AURA_CALIB_ROOT` (default `calibration/`, git-ignored except its README), and the pipeline reads them from there.

| Result | File in `$AURA_CALIB_ROOT` | Written by | Read by |
|---|---|---|---|
| Camera intrinsics, per camera and resolution | `intrinsic_results/1280x720/oak_<MXID>.{npz,json}` (`640x400/` for `low`) | `aura/calibration/Extract_intrinsic.py` | extrinsic and hand-eye tools, every PhysTwin-format conversion, the recording executor's `recording_summary.json` |
| Camera extrinsics (camera-to-world) | `extrinsic_results/oak_<MXID>.{npz,json}` | `aura/calibration/Extract_extrinsic.py` or `aura/calibration/calibrate_3oak_anchor108.py` | `calibrate.pkl`, point-cloud fusion, the grasp filter's camera keep-outs, the hand-eye tool |
| Hand-eye (robot base to world) | `robot_calibration_results/robot_calib_result.{npz,json}` | `aura/calibration/robot_world_calib_charuco.py` | the grasp filter and both executors (the `.npz`) |
| Fingertip geometry | `robot_calibration_results/fingertip_calibration_results/franka_fingertip_table_calibration_simple.json` | copied from `assets/`, or fitted by `aura/calibration/fit_fingertip_translation_from_json.py` | the grasp filter; copied by the lift one-shot into each robot-executed case's `phystwin/` |

Keep the `.npz` and the `.json` of each camera together and consistent. Different consumers read different ones: for example, `aura/capture/anygrasp_data_collection.py` requires an `.npz` for every camera, while `aura/conversion/calibration_matrix_to_phystwin_style.py` prefers the `.json`.

## Before you start

- **Configuration.** In `config/aura.env`, fill in `AURA_OAK_IDS` and `AURA_HANDEYE_OAK_ID` ([hardware.md](hardware.md#cameras)) and, for the robot steps (step 3 and the re-fit in step 4), `AURA_ROBOT_IP` ([hardware.md](hardware.md#robot)). Load the file in every shell:

  ```bash
  set -a; source config/aura.env; set +a
  conda activate "$AURA_ENV"     # camera recordings and all fitting tools; not for the robot steps
  ```

- **Captures.** Write each calibration recording to a new folder under `$AURA_DATA_ROOT/calibration_captures/` (`$AURA_DATA_ROOT` defaults to the git-ignored `data/`). The recorders' `--overwrite` deletes nothing, so old bundles in a reused folder would be mixed into the new capture.
- **Board.** A ChArUco board of 4 × 5 squares of 40 mm with 30 mm `DICT_4X4_50` markers: 160 × 200 mm, 10 markers, 12 inner corners. Print it at exact scale and mount it flat and rigid ([hardware.md](hardware.md#charuco-board) shows how to generate it). Every command below passes this geometry as `--squares-x 4 --squares-y 5 --square-length 0.04 --marker-length 0.03 --dictionary DICT_4X4_50`; `aura/calibration/calibrate_3oak_anchor108.py` has it built in.
- **Undistortion.** Keep `--enable-undistortion True` (the default) in every recorder, now and later. The intrinsics are fitted on frames the camera has already undistorted.
- **Resolution.** Every pipeline stage records at 1280 × 720 (`high`) by default, except the teleoperation recorder of step 3 (default `low`; pass `--resolution high` there), so calibrate 1280 × 720. Add 640 × 400 intrinsics only if you will record with `low`.
- **Gamepad.** The teleoperation recorder (hand-eye session, fingertip contacts) needs a PlayStation-style gamepad. Connect it before you launch: without it the program exits, but only after the arm has homed. Before the first session, check with a joystick test tool that both triggers read -1.0 at rest ([hardware.md](hardware.md#gamepad)); a trigger that rests at 0.0 drives the arm down (L2) or turns it (R2) as soon as teleoperation starts.

## The world frame

The world frame is the frame of the ChArUco board lying flat on the table during the extrinsic calibration. For every camera, `T_world_from_cam` = `T_board_from_cam`, stored as `phys_twin_c2w` and copied into each case's `calibrate.pkl`.

- **Origin:** the outer corner of the board's first square, which is the top-left corner of the image produced by the board generator in [hardware.md](hardware.md#charuco-board).
- **Axes:** +X along the 4-square edge (0.16 m), +Y along the 5-square edge (0.20 m), and +Z = X × Y into the board. With the board face-up on the table, **+Z points down into the table**.
- **Consequences:** the table plane is z = 0 and everything above it has negative z. The cameras sit at negative z, and the robot base frame is flipped relative to the world, so `T_world_from_base[2, 2]` is close to -1. The table filter, the grasp filter and the fused point clouds all use this z-down convention.

## 1. Camera intrinsics

Fit `K` and the residual distortion of each camera, once per capture resolution.

**Record** one sweep per camera. All three cameras record, but only the camera you are calibrating needs to see the board:

```bash
python aura/capture/record_3oak.py --resolution high --duration 30 --preview True \
  --sync-tolerance-ms 75 --output-root "$AURA_DATA_ROOT/calibration_captures/intrinsics_oak_<MXID_0>"
```

Move the board slowly through the whole field of view of that camera: the centre, the edges and the corners, at several distances and with the board tilted in different directions. Hold it close enough: views in which the board covers less than about 10 % of the image (about 270 × 340 px at 1280 × 720, 145 × 180 px at 640 × 400) can be rejected as too small. 30 s is the recorder's maximum (`--duration`). The authors' sweeps gave 87 to 139 accepted views per camera from about 400 to 450 synchronized frames. The `--preview True` window labels each view with its `oak_<MXID>`.

**Fit**, then repeat both steps for `<MXID_1>` and `<MXID_2>`:

```bash
python aura/calibration/Extract_intrinsic.py \
  --root "$AURA_DATA_ROOT/calibration_captures/intrinsics_oak_<MXID_0>" \
  --camera-name oak_<MXID_0> \
  --squares-x 4 --squares-y 5 --square-length 0.04 --marker-length 0.03 \
  --dictionary DICT_4X4_50 --sample-every 1 \
  --output-prefix "$AURA_CALIB_ROOT/intrinsic_results/1280x720/oak_<MXID_0>"
```

- **Method.** The tool detects the markers and interpolates the ChArUco corners in every frame (`--sample-every 1`). It rejects views with fewer than 4 markers or 8 corners, blurry views (Laplacian variance < 60), views in which the bounding box of the ChArUco corners covers less than 3 % of the image (`--min-area-ratio`; the inner corners span only 30 % of the board, hence the 10 % above), and near-duplicates of the previous accepted view. It then calibrates with `calibrateCameraCharucoExtended`. With at least 15 views it re-fits without the worst 20 % (`--prune-worst-fraction`) and keeps the re-fit if its RMS is not higher.
- **Output.** `<prefix>.npz` holds `K` (3 × 3), `dist` (1 × 5), `rms` and `image_size` (`[width, height]`). `<prefix>.json` holds the same values plus the accepted views, filter statistics and per-view errors. `--save-overlay-dir <dir>` additionally writes a keep/drop image per view; the re-fit can still drop a `keep` view, and `accepted_bundle_indices` in the JSON lists the views of the final fit.
- **Check.** RMS reprojection error of about 0.2 to 0.3 px at 1280 × 720, from at least 10 views ([Quality targets](#quality-targets)).
- **Low resolution.** For `low`, record with `--resolution low` and write to `intrinsic_results/640x400/oak_<MXID>`.

> [!IMPORTANT]
> Pass `--sample-every 1`. The default of 30 keeps only every 30th frame and fails with `Accepted only N valid views, need at least 10`. Always pass `--output-prefix` as well: the default, `<root>/<camera-name>_charuco`, is not where the pipeline looks.

## 2. Camera extrinsics

Solve all three cameras against one shared board placement. That placement defines the world frame.

**Place** the board flat on the table in the middle of the workspace, fully visible to all three cameras and not seen too obliquely. Leave it there while you record:

```bash
python aura/capture/record_3oak.py --resolution high --duration 10 --preview True \
  --sync-tolerance-ms 75 --output-root "$AURA_DATA_ROOT/calibration_captures/extrinsics"
```

**Solve** every camera on the same bundle. The scene is static, so any bundle in which all three cameras see the whole board works, for example `bundle_000100`:

```bash
for MXID in $AURA_OAK_IDS; do
  python aura/calibration/Extract_extrinsic.py \
    --bundle-image "$AURA_DATA_ROOT/calibration_captures/extrinsics" "oak_$MXID" bundle_000100 \
    --intrinsic-file "$AURA_CALIB_ROOT/intrinsic_results/1280x720/oak_$MXID.npz" \
    --squares-x 4 --squares-y 5 --square-length 0.04 --marker-length 0.03 --dictionary DICT_4X4_50 \
    --save-overlay "$AURA_CALIB_ROOT/debug/extrinsic/oak_$MXID.jpg" \
    --output-json "$AURA_CALIB_ROOT/extrinsic_results/oak_$MXID.json" \
    --output-npz "$AURA_CALIB_ROOT/extrinsic_results/oak_$MXID.npz"
done
```

`--image-path <capture>/bundle_000100/oak_<MXID>/rgb.jpg --camera-name oak_<MXID>` is equivalent to `--bundle-image`.

- **Method.** The tool detects the board with `cv2.aruco.CharucoDetector`, pairs each detected corner with its 3-D board position and solves the pose with `SOLVEPNP_IPPE`, using the residual distortion stored with the intrinsics. This is the extrinsic convention of the release: calibrate every camera with this tool, and use the re-anchoring tool of [step 5](#5-optional-re-anchor-moved-cameras) only for cameras that moved. The `.npz` and `.json` hold `K`, `dist`, `rvec_cam_from_board`, `tvec_cam_from_board`, `T_cam_from_board`, `T_board_from_cam`, `T_cam_from_world`, `T_world_from_cam`, `phys_twin_c2w`, `image_size`, `num_markers`, `num_charuco` and `reproj_rms`.
- **Check the overlays.** Every camera should find all 10 markers and 12 corners, with an RMS below 0.5 px, and the drawn axes must sit on the board's origin corner.
- **Check the camera poses.** Each camera centre must lie above the table (negative z) and look down into it:

  ```bash
  python - <<'EOF'
  import os
  import numpy as np
  root = os.path.join(os.environ["AURA_CALIB_ROOT"], "extrinsic_results")
  for mxid in os.environ["AURA_OAK_IDS"].split():
      T = np.load(os.path.join(root, f"oak_{mxid}.npz"))["T_world_from_cam"]
      print(f"oak_{mxid}: centre {T[:3, 3].round(3)} m, optical axis z {T[2, 2]:+.2f}")
  EOF
  ```

  Expect a negative z for every centre (the authors' cameras: -0.40 to -0.46 m) and a positive optical-axis z.

> [!IMPORTANT]
> - Pass the intrinsic **`.npz`**. The `.json` report of `Extract_intrinsic.py` has no `image_size` key, so the tool fails with `KeyError: 'image_size'`.
> - **Never pass `--zero-dist`.** The stored `dist` is the residual distortion of already undistorted frames. This tool and the hand-eye tool (step 3) solve with it; only the re-anchoring tool of [step 5](#5-optional-re-anchor-moved-cameras) replaces it with zeros by default.
> - The image size must equal the intrinsics' `image_size` (`Image size mismatch`); use the `1280x720` intrinsics for a `high` capture.
> - Without `--output-json` and `--output-npz`, the results land next to the input image as `<stem>_extrinsic.{json,npz}`, where nothing reads them.
> - Keep only your current cameras' files in `extrinsic_results/`; move old ones into a subfolder such as `backups/`, which the grasp filter does not read. The filter reads every `.npz` directly in that folder whose file name contains `oak_` anywhere (`old_oak_<MXID>.npz` too), and every such `.json` without an `.npz` of the same name. It makes a camera keep-out of each one that holds a camera-to-world transform, and stops if it cannot read one, such as a hidden `._oak_*` file.

## 3. Hand-eye calibration

This step estimates `T_world_from_base`, the pose of the robot base in the world (board) frame. The board, held rigidly in the gripper, is moved through many poses in front of the hand-eye camera `$AURA_HANDEYE_OAK_ID`, which needs steps 1 and 2 first. The robot steps need the real-time kernel, an unlocked robot with FCI active and a clear workspace ([hardware.md](hardware.md#before-every-session)).

### Record a teleoperation session

> [!CAUTION]
> `--home-on-start` defaults to `True`: **the arm moves to its home pose as soon as the program starts, without a prompt.** Pass `--home-on-start False` to skip homing. Check the gamepad first ([Before you start](#before-you-start)).

```bash
conda deactivate; conda deactivate    # leave AuRA and (base): robot scripts run without a conda env
uv run --project environments/robot python aura/robot/gamepad_record_franka_3oak.py \
  --oak-ids <MXID of the hand-eye camera> \
  --resolution high --max-duration 60 --preview True \
  --output-root "$AURA_DATA_ROOT/calibration_captures/handeye"
```

1. Wait until the arm has homed and the console prints `Multiprocess teleop ready.` If the gripper is closed, press D-pad up to open it. Hold the board's tab between the fingers and press D-pad up again to close the gripper on it.
2. Once the console also shows `Press Square to START recording` (the cameras have warmed up), press Square to start recording.
3. Move slowly through many poses. Keep the whole board visible to the hand-eye camera, cover translations of several decimetres and rotations of tens of degrees, and pause briefly now and then.
4. Press Square again, or Circle, to stop; at `--max-duration` the recording stops by itself, and the console prints `Reached max duration`. The recorder then synchronizes and saves the session, and the program exits; every launch records one session. Never stop it with Ctrl-C, which discards the frames held in RAM. Once the recording has stopped, do not press Square until the recorder prints `Done saving.`; Circle is safe ([why](troubleshooting.md#teleoperation-recorder)).

- **Options.** Pass `--resolution high`: the default `low` (640 × 400) does not match the 1280 × 720 intrinsics, and the hand-eye tool would drop the bundles as `drop_exception` (`Image size mismatch`). `--max-duration` defaults to 30 s and is capped at 60 s. The authors' session had 614 synchronized frames, about 40 s of motion. `--preview True` shows the camera's view, so you can keep the board in it. The authors' sessions also passed `--robot-state-hz 30 --sync-tolerance-ms 70`; the command keeps the defaults, 100 Hz (denser robot states) and 45 ms.
- **Cameras.** `--oak-ids` takes bare MXIDs, without the `oak_` prefix. Record only the hand-eye camera, as the authors' sessions did: pass the MXID of `$AURA_HANDEYE_OAK_ID` without its prefix (`--oak-ids "${AURA_HANDEYE_OAK_ID#oak_}"`). Recording every camera works too, but then the hand-eye tool's `--max-match-span-ms` filters on the time span across cameras, and a `--camera-name` that names another recorded camera is used without an error.
- **Defaults.** The robot IP comes from `config/aura.env`, and without `--oak-ids` every camera in `AURA_OAK_IDS` is recorded. Without `--output-root`, the session is written to `./recording_sync_robot_<timestamp>` in the current directory.
- **Releasing the board.** Start the recorder again with `--home-on-start False` and press D-pad up: the first toggle opens a closed gripper.

### Teleoperation controls

| Input (PlayStation layout) | Action |
|---|---|
| Left stick Y / X | translate along x / y |
| Right stick X / Y | roll / pitch |
| L2 (LT) | z down; hold L1 (LB) for z up |
| R2 (RT) | yaw -; hold R1 (RB) for yaw + |
| Square | start / stop recording |
| Circle | stop (saving a running recording) and quit |
| D-pad up | toggle the gripper: open, or grasp at width 0 with 20 N |
| Cross | disabled |

The speed limits are 0.12 m/s and 0.35 rad/s, with a stick deadzone of 0.10. Pressing q or Esc in the preview window ends the session.

### Diagnose, then solve

```bash
conda activate "$AURA_ENV"
python aura/calibration/robot_world_calib_charuco.py \
  --session-root "$AURA_DATA_ROOT/calibration_captures/handeye" \
  --camera-name "$AURA_HANDEYE_OAK_ID" \
  --world-from-camera-file "$AURA_CALIB_ROOT/extrinsic_results/$AURA_HANDEYE_OAK_ID.npz" \
  --intrinsic-file "$AURA_CALIB_ROOT/intrinsic_results/1280x720/$AURA_HANDEYE_OAK_ID.npz" \
  --squares-x 4 --squares-y 5 --square-length 0.04 --marker-length 0.03 --dictionary DICT_4X4_50 \
  --max-robot-delay-ms 30 --prefer-joint-fk \
  --output-dir "$AURA_CALIB_ROOT/robot_calibration_results" \
  --diagnose-only
```

- **`--diagnose-only`** prints, per status, how many bundles were dropped (`drop_robot_delay`, `drop_match_span`, `drop_reproj_rms`, `drop_exception`, ...), with example rows, and how many passed the pose-diversity filter. It writes `scan_records_debug.json` and `pose_filter_logs_debug.json` to `--output-dir`. It exits with status 0 even when every bundle was dropped, so read the final line `prefilter kept N, pose diversity kept M`.
- **Solving.** Run the same command without `--diagnose-only`, and add `--save-final-overlays` to write the board detections of the final samples to `detection_overlays_final/`. `--max-robot-delay-ms 30 --prefer-joint-fk` are the settings recorded in the authors' result; on their session, the defaults (80 ms, no `--prefer-joint-fk`) give the same result to within 0.01 mm.
- **Method.**
  - Each bundle gives the board pose in the world, through the hand-eye camera, and the flange pose, by forward kinematics from `joint_angles_rad` in `meta.json`.
  - A greedy filter keeps a sample only if it differs from the last kept one by at least 2 cm or 5°.
  - `T_ee_from_board` is initialized from relative motions (AX = XB), and `T_world_from_base` by averaging over the samples. Both are refined jointly. If any sample has residuals above 3 cm or 5° and at least 4 others remain, those samples are removed and both are refined again; otherwise the first solve is kept with every sample and no warning (`Final inliers` then equals `Pass pose diversity`, and `outlier_logs` in the JSON still lists the outliers as `drop_final_*`). At least 4 samples must pass the pose-diversity filter.
  - The forward kinematics ends at the flange, without the hand offset. `T_ee_from_board` absorbs every fixed offset between the flange and the board.
- **Output.**
  - `robot_calib_result.npz` holds the five 4 × 4 matrices `T_world_from_camera`, `T_world_from_base`, `T_base_from_world`, `T_ee_from_board` and `T_board_from_ee`.
  - `robot_calib_result.json` adds the counts, most filter settings (`filters`), `error_summary` and per-sample errors. It records `--session-root`, `--world-from-camera-file` and the image paths as given (not `--intrinsic-file`), so do not publish it.
- **Check.**
  - Mean residuals of about 2 mm and 0.7°, and a maximum below 6 mm and 2.5°.
  - `T_world_from_base[2, 2]` close to -1 when the robot stands on the table.
  - `T_world_from_base[:3, 3]` is the robot base origin in board coordinates. Compare it with a tape measure.

> [!IMPORTANT]
> - `--camera-name` (default `$AURA_HANDEYE_OAK_ID`) must be the bundle sub-folder `oak_<MXID>`, with the prefix, and the camera of `--world-from-camera-file`. A wrong name is not caught up front: every bundle is dropped as `drop_exception`.
> - `--intrinsic-file` is required in practice. Without it the tool looks for a `calibration_manifest` that the teleoperation recorder never writes, and fails with `FileNotFoundError`. Pass the `.npz`, not the `.json`.
> - If you recorded several cameras and most bundles are dropped as `drop_match_span`, record again with only the hand-eye camera, or check for a camera on USB 2 ([troubleshooting.md](troubleshooting.md#cameras-and-synchronization)).

## 4. Fingertip geometry

The grasp filter models the gripper by two points: the outer tips of the two fingers, as offsets in the end-effector frame (libfranka's `O_T_EE`, the Franka Hand TCP). It checks them against the table plane at the grasp and along the path.

### Start from the shipped fit

[`assets/franka_fingertip_table_calibration_simple.json`](../assets/README.md) was fitted on the authors' Franka Hand with standard fingers. Its offsets are `[0.001002, ±0.052106, 0.000642]` m, 104.2 mm from outer tip to outer tip with the gripper open. Install it as your starting point:

```bash
mkdir -p "$AURA_CALIB_ROOT/robot_calibration_results/fingertip_calibration_results"
cp assets/franka_fingertip_table_calibration_simple.json \
   "$AURA_CALIB_ROOT/robot_calibration_results/fingertip_calibration_results/"
```

The grasp filter finds it next to `robot_calib_result.npz` (`--calib-results`), and the lift one-shot copies it into every case. Re-fit it if your fingers, pads or end-effector configuration differ.

### Re-fit

1. **Open the gripper fully.** The executor approaches and descends with the gripper open.
2. **Touch the table.** Move the arm, for example with the teleoperation recorder (use `--home-on-start False` after the first launch; the gamepad checks of [Before you start](#before-you-start) apply), until the outer tip of one finger, or of both, just touches the table. Approach slowly: a firm contact can trigger the collision reflex. Vary the orientation between contacts. You need at least 6 contacts; the shipped file has 12 from 9 poses: 3 with both tips, 3 with the right tip only, 3 with the left tip only.
3. **Read the end-effector pose** at each contact. Quit the teleoperation recorder first (Circle), because only one program can hold the arm connection, then run the following from a shell without an active conda env (see [Record a teleoperation session](#record-a-teleoperation-session)):

   ```bash
   uv run --project environments/robot python - <<'EOF'
   import json, os
   import fr3
   with fr3.FR3(ip=os.environ.get("AURA_ROBOT_IP") or None) as robot:
       p = robot.get_pose()
       print(json.dumps({"ee_position_in_base_m": [p.x, p.y, p.z],
                         "ee_orientation_rpy_rad": [p.roll, p.pitch, p.yaw],
                         "joint_angles_rad": robot.get_joints().tolist()}))
   EOF
   ```

   Connecting does not move the arm. The right tip (`right_outer`) is the one on the +y side of the end-effector frame. When only one tip touches, it is the right tip if the end-effector y axis points down, that is, if the third component of the second column of `R = Rz(yaw) Ry(pitch) Rx(roll)` is negative.
4. **Assemble** the contacts into `my_contacts.json`:

   ```json
   {"trajectories": [
     {"sample_id": "sample_01", "observed_contacts": ["right_outer", "left_outer"],
      "ee_position_in_base_m": [0.0, 0.0, 0.0], "ee_orientation_rpy_rad": [0.0, 0.0, 0.0],
      "joint_angles_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
   ]}
   ```

   `observed_contacts` lists the tips that touch, `["right_outer"]`, `["left_outer"]` or both. `joint_angles_rad` is optional provenance.
5. **Fit** (numpy only, for example in the `AuRA` env: `conda activate "$AURA_ENV"`):

   ```bash
   python aura/calibration/fit_fingertip_translation_from_json.py my_contacts.json -o my_fit.json
   ```

   The model is symmetric (right = `[x, +y, z]`, left = `[x, -y, z]`) and assumes all contacts lie on one plane in the base frame. The fitter alternates a plane fit and a linear least-squares fit of the offsets, from 120 starting guesses.
6. **Install.** The pipeline file keeps your observations next to the fit. The fitter's output has no `trajectories` block, so never pass the pipeline file as `-o`: that would delete your observations and make a later re-fit impossible. Write it as follows instead:

   ```bash
   python - <<'EOF'
   import json, os
   contacts = json.load(open("my_contacts.json"))
   fit = json.load(open("my_fit.json"))
   out = os.path.join(os.environ["AURA_CALIB_ROOT"], "robot_calibration_results",
                      "fingertip_calibration_results", "franka_fingertip_table_calibration_simple.json")
   os.makedirs(os.path.dirname(out), exist_ok=True)
   with open(out, "w") as f:
       json.dump({"trajectories": contacts["trajectories"],
                  "end_effector_fingertip_translation": fit["end_effector_fingertip_translation"]}, f, indent=2)
   print("wrote", out)
   EOF
   ```

- **Check.** In `my_fit.json`, `residual_summary.max_abs_residual_mm` should be below 1 mm, and `end_effector_fingertip_translation.outer_to_outer_distance_mm` about 104 mm for open standard fingers.
- **Pitfalls.**
  - Downstream code reads only `right_outer_tip_in_ee_m` and `left_outer_tip_in_ee_m`.
  - Keep the file name and folder. The filter looks for `fingertip_calibration_results/franka_fingertip_table_calibration_simple.json` in the folder of `robot_calib_result.npz`, and otherwise takes the first `*fingertip*calibration*.json` below it.

## 5. Optional: re-anchor moved cameras

Use `aura/calibration/calibrate_3oak_anchor108.py` when one or both of the other cameras moved but the hand-eye camera did not. It measures all three cameras on one new board placement and rebases them so that the anchor camera keeps its current world pose. The world frame and `robot_calib_result.npz` therefore stay valid. If the anchor camera moved, redo steps 2 and 3 instead.

```bash
python aura/capture/record_3oak.py --resolution high --duration 10 --preview True \
  --sync-tolerance-ms 75 --output-root "$AURA_DATA_ROOT/calibration_captures/reanchor"
python aura/calibration/calibrate_3oak_anchor108.py \
  --session-root "$AURA_DATA_ROOT/calibration_captures/reanchor" \
  --bundle bundle_000100 --output-dir "$AURA_CALIB_ROOT/reanchor/$(date +%Y%m%d_%H%M%S)"
```

- **Placement.** The board can lie anywhere all three cameras see it; it does not have to be at the world origin.
- **Inputs.** The anchor is `--anchor oak_<MXID>` (default `$AURA_HANDEYE_OAK_ID`). The tool reads `intrinsic_results/1280x720/oak_<MXID>.npz` for every camera and the anchor's current extrinsic from `$AURA_CALIB_ROOT`.
- **Output.** It writes `oak_<MXID>.{json,npz}` and `oak_<MXID>_board_overlay.png` for each camera to `--output-dir`, which must be new or empty and must not be `extrinsic_results/`. Each JSON records the anchor it used in `anchor_camera` and carries the fixed label `"world_frame": "existing_world_preserved_via_oak108"`. The command above uses a new timestamped folder under `$AURA_CALIB_ROOT/reanchor/`, which is git-ignored with the rest of `calibration/`. Keep these results there or outside the repository: the JSON files record absolute paths and your camera MXIDs. The tool stops if a camera has fewer than 8 corners or 4 markers, or an RMS above `--max-rms` (1.0 px).
- **Install.** Review the overlays in the folder printed as `[OUTPUT]`, then run the same `calibrate_3oak_anchor108.py` command again with `--apply` (the timestamp gives it a new empty `--output-dir`). This backs up the two non-anchor extrinsics to `extrinsic_results/backups/before_anchor108_<timestamp>/` and installs the new ones. The `108` in this folder name, in the `world_frame` label and in the tool's name is the authors' historical name for their own anchor camera, not yours. The tool verifies that the anchor extrinsic and `robot_calib_result.{npz,json}` were not changed.
- **Differences from `Extract_extrinsic.py`.** The board (4 × 5, 40 / 30 mm) and the 1280 × 720 intrinsics are built in. By default it solves with zero distortion and `SOLVEPNP_ITERATIVE`; `--raw-distorted` uses the stored residual distortion instead. The command above uses the default, as the authors' re-anchoring did, so the re-anchored cameras follow the zero-distortion convention, while the anchor camera keeps its pose from step 2. On the authors' capture, the two tools differ by 4 to 5 mm and 0.3 to 0.5° for the same image. `Extract_extrinsic.py` is the reference convention; a full recalibration with it (steps 2 and 3) is preferable to re-anchoring whenever the hand-eye camera may have moved as well.

## After calibrating

1. **Start new cases.** Each case keeps copies of the calibration it was recorded with. Do not run the lift one-shot on a case prepared or recorded before this calibration. Its preparation reuses the case's capture, point cloud and grasps from the old calibration and re-runs only the grasp filter (default `--refilter`), so on a case not yet recorded the robot executes grasps computed with the old calibration. On a recorded case the run ends with `Refusing to overwrite different preserved file`, but by then it has rewritten the case's safe list, `calibrate.pkl` and `metadata.json` with the new calibration ([troubleshooting.md](troubleshooting.md#execution-and-lift-recording) shows the repair).
2. **Run the preparation** on a test object, with the arm out of the cameras' view, for example at home. To home the arm before your first case, use the teleoperation recorder (the hover-test command that [usage.md](usage.md#dry-run-and-hover-only-test) uses for this needs a case's safe list): from a shell without an active conda env (see [Record a teleoperation session](#record-a-teleoperation-session)), run `uv run --project environments/robot python aura/robot/gamepad_record_franka_3oak.py`. It homes the arm on launch, without a prompt (the gamepad checks of [Before you start](#before-you-start) apply). Once the console prints `Multiprocess teleop ready.`, press Circle, not Square: the program quits without recording. Then, with the `AuRA` env active (`conda activate "$AURA_ENV"`), run `bash scripts/anygrasp_poses_one_shot.sh <case>`. Stage 04 should report `Camera keep-outs: 3 cameras` and accept some grasps.
3. **Dry run and hover test.** Do a dry run of the recording executor, which checks the calibration and the stored hover pose without connecting to anything. Then do a hover-only test with the **standalone** executor and check that the TCP, between the fingertips, stops 10 cm (`--hover-z-offset`) straight above the grasp centre. This is the end-to-end check of `T_world_from_base`. The grasp centre is where the fingertips close, usually below the object's top, so the fingertips stop less than 10 cm above the object. Both commands are in [usage.md](usage.md#dry-run-and-hover-only-test). If the standalone executor rejects the target as outside its workspace box, see [troubleshooting.md](troubleshooting.md#execution-and-lift-recording).

## Quality targets

| Step | Check | Target | Authors' rig |
|---|---|---|---|
| Intrinsics, 1280 × 720 | RMS reprojection error | 0.2 to 0.3 px, ≥ 10 views | 0.236 / 0.257 / 0.254 px, 87 to 139 views |
| Intrinsics, 640 × 400 | RMS reprojection error | similar | 0.227 / 0.210 / 0.167 px |
| Extrinsics | markers, corners, RMS | 10, 12, < 0.5 px | 10, 12, 0.310 / 0.268 / 0.191 px |
| Hand-eye | residual mean, max | about 2 mm and 0.7°; < 6 mm and 2.5° | 2.23 mm, 0.65°; 5.78 mm, 2.32° (median 1.94 mm, 0.49°; of 614 bundles, 585 passed detection and sync checks, 119 the pose-diversity filter, 119 were inliers) |
| Fingertips | largest plane residual | < 1 mm | 0.94 mm (mean 0.41 mm, 12 contacts) |
