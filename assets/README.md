# `assets/` — files shipped with the repository

Both files were measured on the authors' Franka Research 3. They are valid for your
setup only under the conditions stated below; otherwise regenerate them.

| File | What it is | Read by |
|---|---|---|
| `fr3_home_state.npz` | FR3 joint configuration at the `fr3` `HOME_Q` pose | grasp filter (`--state-file`), lift one-shot |
| `franka_fingertip_table_calibration_simple.json` | outer-fingertip offsets of the Franka Hand: fitter input **and** fit result read by the pipeline | grasp filter, lift one-shot |

## `franka_fingertip_table_calibration_simple.json`

This one file is both the input of
[`aura/calibration/fit_fingertip_translation_from_json.py`](../aura/calibration/fit_fingertip_translation_from_json.py)
and the calibration result the pipeline consumes:

- **`trajectories` (fitter input):** 9 table-touch poses with 12 contact observations
  (3 with both outer fingertips on the table, 3 right only, 3 left only). Each entry holds
  `sample_id`, `observed_contacts`, `ee_position_in_base_m` and `ee_orientation_rpy_rad`
  (end-effector pose in the robot base frame; RPY convention as in `fr3.FR3().get_pose()`),
  plus provenance fields the fitter does not read (`contact_type` is only a fallback for
  missing `observed_contacts`; `joint_angles_rad`, `plane_residuals_m`, `plane_residuals_mm`).
- **`end_effector_fingertip_translation` (fit result, read by the pipeline):** the outer-fingertip offsets in
  the end-effector frame, symmetric by construction (right = `[x, +y, z]`, left = `[x, -y, z]`):
  `right_outer_tip_in_ee_m` = `[0.001002, 0.052106, 0.000642]` m, i.e. 104.2 mm outer to outer.
  The `*_mm` keys repeat the values in millimetres. Downstream code reads only
  `right_outer_tip_in_ee_m` and `left_outer_tip_in_ee_m`.

The offsets describe the authors' gripper fingers and assume the robot's configured end
effector is the stock Franka Hand (the grasp filter applies them at PyRoki's `fr3_hand_tcp`).
Re-fit them if your fingers, pads or end-effector settings differ.

**Install.** The grasp filter (`aura/grasping/filter_grasps_table_collision_pyroki.py`)
looks for the file under the directory that holds your hand-eye result
`robot_calib_result.npz`, so copy it there. Run this from the repository root, after `set -a; source config/aura.env; set +a`, which sets `$AURA_CALIB_ROOT` (default `calibration/`):

```bash
mkdir -p "$AURA_CALIB_ROOT/robot_calibration_results/fingertip_calibration_results"
cp assets/franka_fingertip_table_calibration_simple.json \
   "$AURA_CALIB_ROOT/robot_calibration_results/fingertip_calibration_results/"
```

`scripts/run_anygrasp_to_lift_one_shot.sh` copies it from there into every recorded case.

**Re-fit** (pure numpy; the fit itself needs no robot):

```bash
python aura/calibration/fit_fingertip_translation_from_json.py my_contacts.json -o my_fit.json
```

The fitter's output has no `trajectories` block, so never pass the input file as `-o`:
that would delete the observations. Write to a new file, then install your contacts and the fit
together as in [docs/calibration.md, Re-fit, step 6](../docs/calibration.md#re-fit). Copying
only the `end_effector_fingertip_translation` block into an installed copy of this file would
keep the authors' `trajectories` next to your fit. Re-running the shipped fitter on the
shipped trajectories gives offsets within 0.031 mm of the stored block (the fitter stops after
100 iterations per start; the stored block is the fully converged fit of the same model).

## `fr3_home_state.npz`

The FR3 arm joint configuration at the `fr3` wrapper's `HOME_Q` pose
(`[0, -π/4, 0, -3π/4, 0, π/2, π/4]`, `third_party/fr3/src/fr3/_types.py`), measured on the
authors' robot on 2026-08-17 after homing, from 20 joint-state samples
(`q` differs from `HOME_Q` by at most 0.0007 rad).

| Key | Content |
|---|---|
| `q` | mean of the samples, shape `(7,)`, rad |
| `q_std` | per-joint standard deviation of the samples, `(7,)` (1.7e-6 to 4.0e-6 rad) |
| `samples` | the 20 measured joint vectors, `(20, 7)` |
| `source` | `"measured_after_fr3_home"` |
| `created_at` | `"2026-08-17T21:55:07.799593"` |

It is the start state of the PyRoki grasp filter: the one-shot scripts pass it as
`--state-file` by default, and the filter plans the motion from `q` (the only key it reads).
This is valid because the recording executor homes the robot before it moves (default `--home` of
`scripts/run_anygrasp_to_lift_one_shot.sh`). With `--no-home` the script requires an explicit,
freshly captured `--state-file`, given as an absolute path. On a prepared case, also keep the
default `--refilter`: `--no-refilter` keeps the safe list planned from the old state file. The executors
themselves home only when you pass `--home`, which is off by default. When you run one by
hand with `--execute`, pass `--home` unless the filter's `--state-file` was captured at the
arm's current pose. If your robot settles at a different home configuration, or `HOME_Q`
changes, capture a new file with key `q`.

No script in this repository generates this file.
