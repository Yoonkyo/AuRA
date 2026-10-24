# Usage

This page is the runbook for one data-collection session, from the pre-flight checklist to a verified grasp-and-lift recording, plus the camera-only human-baseline recording. It assumes a complete [installation](installation.md) and a calibrated rig ([calibration.md](calibration.md): intrinsics, extrinsics, hand-eye and the fingertip offsets). All commands run from the repository root. The tools are named by their role: the **preparation script** `scripts/anygrasp_poses_one_shot.sh` (Step 1), the **lift one-shot** `scripts/run_anygrasp_to_lift_one_shot.sh` (Step 2), the **recording executor** `aura/robot/execute_anygrasp_franka_record_lift.py` (the lift one-shot's robot step and the dry run) and the **standalone executor** `aura/robot/execute_anygrasp_franka.py` (the hover test).

## Pre-flight checklist

Before every session:

- [ ] Real-time: `cat /sys/kernel/realtime` prints `1`, `ulimit -r` prints `99`, and `ulimit -l` prints `unlimited`.
- [ ] Robot: powered on, workspace clear, emergency stop within reach. In Franka Desk (`https://<ROBOT_IP>/desk/`), the joints are unlocked and **FCI is activated**. `ping -c 3 <ROBOT_IP>` answers.
- [ ] Cameras: `lsusb -d 03e7: | wc -l` prints `3`, every camera is on USB 3, and none has moved since calibration.
- [ ] GPU: `nvidia-smi` shows no leftover process.
- [ ] Disk: `df -h` shows enough free space on the drive that holds `$AURA_DATA_ROOT` (default `data/`), about 0.6 GB per robot-executed case ([hardware.md](hardware.md#workstation)). The lift recording stays in RAM until the lift has ended, so a full disk loses it after the robot has grasped and lifted the object.
- [ ] Files: the calibration, including the fingertip file, is in `$AURA_CALIB_ROOT` (default `calibration/`), the weights are in `$AURA_GSAM_CHECKPOINTS` (default `checkpoints/groundedSAM/`), and the AnyGrasp SDK, license and checkpoint are in `$AURA_ANYGRASP_SDK`.
- [ ] Shell: run `conda activate AuRA` (or the env you named in `AURA_ENV`), because the one-shot scripts call `python` for their own bookkeeping. `conda` and `uv` must be on `PATH`. For the commands below, also run `set -a; source config/aura.env; set +a`.
- [ ] Display: work in a desktop session on the workstation, not a plain SSH shell. Stage 03 renders its debug images through a hidden Open3D window and crashes without a display ([troubleshooting.md](troubleshooting.md#stage-03-anygrasp)). The camera previews also open windows.

## Step 1: Preparation (no robot motion)

Place the object on the table inside the calibrated workspace, with the arm out of the cameras' view (for example at home), and run:

```bash
CASE=green_toy_1
bash scripts/anygrasp_poses_one_shot.sh "$CASE"
```

- **Case names** match `[A-Za-z0-9_-]+`. The Grounded-SAM-2 text prompt is derived from the name: a trailing `_<digits>` is dropped and underscores become spaces, so `green_toy_1` becomes `"green toy."`. Override it with `--sam-prompt "green toy."`.
- **Stages:** `01_collect_anygrasp_frame` (one synchronized three-camera snapshot), `02_generate_observation_pointcloud`, `03_anygrasp_pose_generation`, and `04_pyroki_filter`. If AnyGrasp finds no grasp, stage 03 retries automatically without collision detection as `03b_...`. Each stage logs to `<case>/logs/<stage>.log`.
- **Location:** the case is written to `$AURA_DATA_ROOT/franka_lift/<case>/`. `--data-root DIR` writes it to `DIR/<case>/` instead; then pass the same `--data-root` to the preparation script and the lift one-shot, and replace `$AURA_DATA_ROOT/franka_lift` with `DIR` in the commands below. To move every case, set `AURA_DATA_ROOT` in `config/aura.env` instead; the commands below then work unchanged.
- **Re-runs:** an existing case is an error unless you pass one of these:
  - `--resume` skips each stage whose own outputs exist, even if an earlier stage runs again, and runs the others ([Re-runs](troubleshooting.md#re-runs)). Add `--refilter` to re-run only the filter, for example after changing a threshold.
  - `--force` deletes the preparation outputs, including the static capture in `anygrasp/` and the stage logs in `logs/`, captures the scene again and keeps `phystwin/`. Never use it on a case that already has a lift recording (`phystwin/raw_recording/`): the new capture would show the scene after the lift, and the pre-interaction capture is lost for good. Use a new case name instead.
- **Options:** common ones are `--top-k 30` (more AnyGrasp candidates), `--preview` (live camera window during the capture) and `--mask-object-id union` (fuse every mask component, not only the largest). `bash scripts/anygrasp_poses_one_shot.sh --help` lists every option with its default.
- **Filter thresholds of the paper.** The cases recorded for the paper were filtered with `--clearance-m 0.005 --robust-joint-margin-rad 0.2`, which are stricter than the preparation script's defaults (0.003 m of fingertip clearance above the table, 0.16 rad of joint-limit margin along the path). To use the paper's values, add them to the command, for example `bash scripts/anygrasp_poses_one_shot.sh --clearance-m 0.005 --robust-joint-margin-rad 0.2 "$CASE"`, or re-run only the filter of an existing case with `--resume --refilter`. With the lift one-shot, pass them again after `--`, also when Step 1 already used them ([Step 2](#step-2-full-autonomous-run)).

Do not touch the object or the cameras between preparation and execution: the grasp is computed from this snapshot. To re-stage a scene from an earlier case, `python aura/capture/manual_align_3oak.py --overlay-reference-root "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp"` (AuRA env, cameras only) overlays that case's static snapshot on the live images. The reference must be a folder with `color/<cam>/` images or raw `bundle_*` folders, such as `<case>/anygrasp`, `<case>/phystwin` (its frame 0 is the lift onset, with the object in the gripper) or a human-baseline case; the case folder itself is rejected.

### Inspect the grasps

```bash
python - "$AURA_DATA_ROOT/franka_lift/$CASE/anygrasp_results/grasps_table_ik_report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps(r["summary"], indent=2))
for e in r["entries"]:
    if not e["accepted"]:
        print(e["rank"], e["rejection_stage"], e["reason"])
PY
```

- **Report:** `summary` gives the number of AnyGrasp candidates accepted and the rejection counts per stage. Only the first failing check of each grasp is counted. For comparison, the authors' 26 cases, filtered with the paper's thresholds (`--clearance-m 0.005 --robust-joint-margin-rad 0.2`), each kept 1 to 10 of their 2 to 20 candidates (105 of 354 in all; 7 cases kept a single grasp); about three quarters of the rejections were at `home_to_hover_cartesian_robustness` and most of the rest at `target_fingertip_table`. If no grasp survives, see [troubleshooting.md](troubleshooting.md#stage-04-feasibility-filter).
- **Safe list:** the accepted grasps are in `anygrasp_results/grasps_table_ik_safe.json`, sorted by total joint motion. Index 0 is therefore not necessarily AnyGrasp's best-scored grasp.
- **Renders:** look at all of these.
  - `anygrasp_results/debug_all_grasps.png` and `debug_best_grasp.png` show the AnyGrasp candidates before filtering, rendered as in the AnyGrasp demo. If the no-collision retry (stage 03b) ran, they are in `anygrasp_results_no_collision/` instead; those candidates were proposed without collision detection, so inspect them with extra care.
  - `anygrasp/mask/cam<N>_frame0_sam_rgba.detections.jpg` shows the detection box chosen in each camera.
  - `anygrasp/observation/visualization/` and `observation_frame000_filtered.ply` show the fused object point cloud.
- **The feasibility filter is not a full collision checker.** It checks fingertip-table clearance, camera keep-out and IK along the path; the arm links, the hand body, self-collision, the object and the camera mounts are not modelled. Judge the scene and the renders yourself (see [Known limitations](troubleshooting.md#known-limitations)).

## Dry run and hover-only test

A **dry run** of the recording executor connects to neither the robot nor the cameras. For the selected grasp, it checks the calibration, the pose mapping and the hover transform stored by the filter. Run it, and the hover test below, from a shell without an active conda env (`conda deactivate`, twice if the shell started in `(base)`):

```bash
CASE_DIR="$AURA_DATA_ROOT/franka_lift/$CASE"
uv run --project environments/robot python aura/robot/execute_anygrasp_franka_record_lift.py \
  --calib "$AURA_CALIB_ROOT/robot_calibration_results/robot_calib_result.npz" \
  --grasp-json "$CASE_DIR/anygrasp_results/grasps_table_ik_safe.json" \
  --rank 0 --pose-frame world --orientation-mode anygrasp --rot-offset-rpy 0 1.57079632679 0 \
  --record-output-root "$CASE_DIR/phystwin/raw_recording"
# ends with: [dry-run] Robot was NOT connected. ...
```

`--rank` is the index into the safe list; the lift one-shot calls the same index `--safe-index`. When you run the executor by hand, always pass `--record-output-root`.

If your table lies below the plane the robot base is mounted on, the output also shows `target workspace check: BAD  z=... outside [+0.00, +0.60]`. This does not stop the dry run: the recording executor does not enforce that workspace box ([Known limitations](troubleshooting.md#known-limitations)). The standalone executor of the hover test below does.

For a **hover-only test**, the robot homes, opens the gripper, moves to 10 cm above the grasp and stops. Use the **standalone** executor for this; the recording executor hangs with `--hover-only` (see [Known limitations](troubleshooting.md#known-limitations)):

```bash
uv run --project environments/robot python aura/robot/execute_anygrasp_franka.py \
  --calib "$AURA_CALIB_ROOT/robot_calibration_results/robot_calib_result.npz" \
  --grasp-json "$CASE_DIR/anygrasp_results/grasps_table_ik_safe.json" \
  --rank 0 --pose-frame world --rot-offset-rpy 0 1.57079632679 0 \
  --execute --home --hover-only
```

> [!CAUTION]
> The standalone executor asks for confirmation (type `yes`; any other input, Enter included, aborts) before it connects, homes and moves to the hover pose; the gripper opens after homing without a prompt. After a reflex (for example a collision) during a Cartesian move, such as the move to hover, the `fr3` wrapper recovers automatically and restarts the move from where the arm stopped towards the same target, up to 3 more times, each with a 1.5 times longer duration and without a prompt; homing and the recording executor do not retry. **Keep a hand on the emergency stop and press it at the first contact:** a stop is not retried.

Differences from the recording executor:

- It does not read the checks the filter stored in the safe list: no robustness check and no 1 mm / 0.5° hover cross-check, and it also accepts an unfiltered `grasps.json`. It moves to whatever pose its arguments give, so use the same `--calib`, `--rank`, `--rot-offset-rpy 0 1.57079632679 0` and z offsets as in the dry run above. Its own `--rot-offset-rpy` default is no offset: without the flag nothing warns you, and the hover pose is turned by 90°.
- It moves with the `fr3` wrapper, not the recording executor's direct Cartesian loop, so the test shows the hover *pose*, not the exact path.
- Its `--motion-speed` and `--descend-speed` are `fr3` speed fractions, not m/s.
- It enforces a Cartesian workspace box in the robot base frame (defaults x 0.20 to 0.80 m, y -0.40 to 0.40 m, z 0 to 0.60 m). If your table lies below the plane the robot base is mounted on, grasp targets have a negative base z (about -0.02 m on the authors' rig) and the guard rejects them with `target outside workspace`; after checking the target, lower `--workspace-z-min` (see [troubleshooting.md](troubleshooting.md#execution-and-lift-recording)).

On the case prepared above, the lift one-shot only re-runs the filter before its robot step, which homes the arm first (`--home`, the default), so you can leave the arm at the hover pose. A run that captures the scene does not home the arm first: Step 1, and the lift one-shot on a new case or with `--force-preparation` or `--overwrite`. Before those, move the arm out of the cameras' view, for example by running the hover-test command again and answering `no` at `Move to HOVER pose?` once the arm has homed: the static snapshot must show the object without the arm.

## Step 2: Full autonomous run

Run this from a shell with the `AuRA` env active (see the pre-flight checklist):

```bash
# interactive: press Enter at every robot checkpoint (recommended for each new object, scene or calibration)
bash scripts/run_anygrasp_to_lift_one_shot.sh "$CASE"

# no checkpoints: only once you trust your calibration and scene; keep the emergency stop within reach
bash scripts/run_anygrasp_to_lift_one_shot.sh --yes "$CASE"
```

- **Checkpoints:** without `--yes` (or `-y`), the run waits for Enter before connecting, homing, moving to hover, descending, closing the gripper and lifting, and again before the final release-and-home step. Enter (or y/yes, case-insensitive) continues; any other input aborts. The gripper opens without a checkpoint, after homing and before the hover move. Before you answer the first checkpoint, check that the printed `Original AnyGrasp rank` equals the AnyGrasp rank that your dry run and hover test printed as `rank=` (not the `--rank` you passed).
- **Re-filter:** on an existing case, such as the one from Step 1, only the filter (stage 04) runs again (default `--refilter`), with the preparation defaults plus the options after `--`, not the filter options of Step 1. It overwrites the `grasps_table_ik_safe.json` and the report you inspected, so safe index 0 can be a different grasp, or the run stops with `PyRoki filtering accepted no grasps`. Repeat after `--` every threshold you changed in Step 1, or, with the default `--home`, pass `--no-refilter` to execute the safe list you inspected as it is; the executor still refuses it if the hover pose changed by more than 1 mm or 0.5°.
- **Stages:**
  1. `01_prepare_and_filter` runs the preparation script: a new case is captured from scratch; an existing case is resumed and re-filtered (see Re-filter above).
  2. `02_franka_grasp_lift_record`: home, hover, descend, grasp and lift; only the lift is recorded.
  3. `02b_release_and_return_home`.
  4. `03_convert_dynamic_to_phystwin`, `03b_generate_mp4_previews` and `04_generate_dynamic_camera_calibration`.
  5. Finally, the script copies the calibration files it used and writes `global_one_shot_summary.json`.
- **Logs:** the robot step logs only to the console. To keep a log, run `mkdir -p logs && bash scripts/run_anygrasp_to_lift_one_shot.sh "$CASE" 2>&1 | tee "logs/$CASE.log"`. The checkpoints still read from the terminal. The log records absolute paths and the camera MXIDs, and usually the robot address; see [Before sharing data](data_format.md#before-sharing-data).

**Conservative first run.** Use slower motions (defaults: 0.10, 0.02 and 0.05 m/s for hover, descent and lift):

```bash
bash scripts/run_anygrasp_to_lift_one_shot.sh --safe-index 0 \
  --motion-speed 0.05 --descend-speed 0.015 --lift-speed 0.03 "$CASE"
```

- **Speeds** are average linear speeds in m/s. The minimum-jerk profile peaks at 1.875 times that value, and a move can be slower when its rotation dominates.
- **Joint margin:** `--cartesian-runtime-joint-margin-rad` (default 0.15 rad) aborts a move before any joint comes that close to a soft limit. Raising it is more conservative, but it can abort grasps that the filter accepted with its 0.16 rad path margin.

**Other options.** `--help` lists all of them.

- `--safe-index N` executes another accepted grasp.
- Options after `--` go to the preparation script, for example `... "$CASE" -- --camera-keepout-radius-m 0.15`. On an existing case only the filter runs again (see Re-filter above), so capture, segmentation and AnyGrasp options (`--preview`, `--mask-object-id`, `--top-k`, ...) take effect only for a new case or with `--force-preparation`, which captures again (never on a recorded case). The offsets (`--hover-z-offset`, `--final-z-offset`, `--lift-z-offset`, `--rot-offset-rpy`), `--state-file`, `--data-root` and `--robot-calib` (the NPZ file; the lift one-shot passes its folder to the preparation script as `--robot-calib-results`, which hands it to the filter as `--calib-results`) are options of the lift one-shot itself and go before `--`.
- By default the object is released where the lift ended and the arm returns home. `--no-release-after` keeps the object in the gripper. Release it yourself before the next robot run (for example from Franka Desk), because the hover test and the lift one-shot open the gripper after homing without a checkpoint.
- `--overwrite` deletes the **entire** case, including `phystwin/`, without asking, and starts again from a new capture. `--force-preparation` rebuilds the preparation outputs from a new capture; on a recorded case it replaces the pre-interaction capture with one taken after the lift and skips the robot step, so use a new case name instead.
- If a complete raw recording already exists, the robot step is skipped and no robot IP is needed; `--finalize-existing-recording` still releases the object and homes the arm, which needs the IP. Re-run or finalize a recorded case with `--no-refilter` and the recorded run's options, which are under `args` in `phystwin/raw_recording/recording_summary.json`: `--safe-index` (`rank`), the speeds, `--record-resolution` (the conversion takes the intrinsics from the value you pass) and `--record-fps`. Without `--no-refilter`, stage 01 first re-filters and rewrites `grasps_table_ik_safe.json`, which a later `--no-refilter` cannot undo, and the run stops before the release and the conversion if the new list is empty or shorter than `--safe-index` + 1.

## Verify the recording

```bash
python - "$AURA_DATA_ROOT/franka_lift/$CASE/phystwin/recording_summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); o = s["offline_sync"]
print("saved", o["num_saved_timesteps"], "of", o["num_target_timesteps"], "dropped", o["num_dropped_timesteps"])
print("match span ms p50/p95", o.get("match_span_ms_p50"), o.get("match_span_ms_p95"))
print("robot states", s["robot_states"]["num_robot_states"], "| reader errors", s["reader_errors"])
PY
```

- **Typical values:** with the default settings, the authors' 26 lift recordings had 41 to 48 target timesteps, 355 to 359 robot states and no reader errors.
- **Dropped timesteps:** in 25 of these recordings one camera delivered fewer frames, and 18 to 33 % of the timesteps were dropped (`match_span_ms_p50` 53 to 77 ms in all but two), although every camera had reported `UsbSpeed.SUPER` (USB 3) in the static capture of the same case (`<case>/logs/01_collect_anygrasp_frame.log`); the other recording dropped none of its 48. Far more drops, or a camera whose `USB speed=` line at start says `UsbSpeed.HIGH` (USB 2), point to a bad link: fix the connection ([hardware.md](hardware.md#cameras)) instead of raising the sync tolerance.
- **Videos:** watch `phystwin/color/0.mp4`, `1.mp4` and `2.mp4`.
- **Aborted lifts:** an aborted lift still produces a normal-looking recording, so also check the console output for `[robot] ERROR`.

## Outputs

```
$AURA_DATA_ROOT/franka_lift/<case>/
├── anygrasp/                       static pre-interaction capture (PhysTwin format, one frame)
│   ├── color/ depth/ calibrate.pkl metadata.json
│   ├── mask/                       Grounded-SAM-2 masks, detection overlays
│   └── observation/                observation_frame000_filtered.{npz,ply} (AnyGrasp input), visualization/
├── anygrasp_results/               grasps.json, grasps_table_ik_safe.json, grasps_table_ik_report.json, debug_*.png
├── anygrasp_results_no_collision/  only if the no-collision AnyGrasp retry ran: its grasps.json and debug_*.png
├── logs/                           one log per preparation stage
├── _work/                          scratch of the static capture
├── phystwin/                       the grasp-and-lift recording (PhysTwin format)
│   ├── color/<cam>/<frame>.png  color/<cam>.mp4  depth/<cam>/<frame>.npy  source_meta/<frame>.json
│   ├── calibrate.pkl  metadata.json  conversion_metadata.json  camera_calibration/
│   ├── robot_calib_result.npz  fr3_home_state.npz  franka_fingertip_table_calibration_simple.json
│   ├── raw_recording/              synchronized raw bundles: bundle_<NNNNNN>/oak_<MXID>/{rgb.jpg,depth.npy}, meta.json
│   └── recording_summary.json  session_summary.json   copies from raw_recording/
├── one_shot_summary.json           preparation parameters ("robot_motion_commanded": false)
└── global_one_shot_summary.json    execution and recording options of the last complete call, including its safe_index
```

A call that reuses an existing recording also rewrites `global_one_shot_summary.json` with its own options; the executed index and speeds are `rank`, `motion_speed`, `descend_speed` and `lift_speed` under `args` in `phystwin/recording_summary.json`.

Each case keeps copies of the calibration files it was converted with. Do not run the lift one-shot on a case prepared or recorded before a recalibration: on a case not yet recorded, the robot executes grasps computed with the old calibration ([After calibrating](calibration.md#after-calibrating)). On a recorded case the run stops, but only after it has rewritten the safe list and its report (default `--refilter`) and rebuilt `calibrate.pkl` and `metadata.json` from the new calibration ([troubleshooting.md](troubleshooting.md#execution-and-lift-recording)). Post-processing these cases is part of v2.

## Human-baseline collection

To record a person manipulating the object with the three cameras only (no robot):

```bash
bash scripts/collect_hand_phystwin_one_shot.sh colorful_rope_1
```

1. Wait for `[READY] Recording is active`: an Enter typed earlier stops the recording as soon as it starts. Keep the first frame clear, then manipulate the object.
2. Press **Enter** once to stop. The recording also stops on its own after 30 s; `--max-duration` can only shorten that. Never press Ctrl-C, which loses the take.
3. After the stop, type nothing in the terminal until the frame prompt appears: typed keys answer the prompts, and after a stop by the time limit or by q or Esc in the preview window, an Enter aborts the save (`Recorder failed with exit code 130`; [troubleshooting.md](troubleshooting.md#human-baseline-recorder)).
4. Review the numbered preview `raw_preview_3oak_numbered.mp4`.
5. Enter the start and end frame, both inclusive, or pass `--start-frame N --end-frame M`; write the numbers without leading zeros.

The case is written in PhysTwin format to `$AURA_DATA_ROOT/hand_collected/<case>/`, together with `raw_recording/` and `hand_collection_summary.json`. It contains no robot state. The script needs an interactive terminal and cannot resume; `--overwrite` replaces an existing case, and `--no-preview` turns off the live camera window.

## Teleoperation recording for calibration

Hand-eye calibration needs a gamepad-driven session with the ChArUco board held in the gripper, recorded by the hand-eye camera. The command, the controls, the `--home-on-start` caution and the camera choice are in [calibration.md](calibration.md#record-a-teleoperation-session).

## Data format

Every case uses the PhysTwin input format: `color/<cam>/<frame>.png`, `depth/<cam>/<frame>.npy` (`uint16` millimetres), `calibrate.pkl`, `metadata.json` and `source_meta/<frame>.json`, which holds the sync provenance of each frame and, for robot cases, the robot state interpolated to the frame time. The full specification of everything the pipeline writes, including the raw bundles, the grasp JSON files and the summaries, is in [data_format.md](data_format.md).
