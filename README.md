<h1 align="center">AuRA: From Atoms to Bits through Autonomous Robot Interaction</h1>

<p align="center">
  🌐 <a href="https://aura-robot.github.io/">Project Page</a> ·
  📄 <a href="https://arxiv.org/abs/2610.34452/">Paper</a> ·
  🎬 <a href="https://youtu.be/z5mpP75-XB0">Video</a>
</p>

<p align="center">
  Xinchen Yu<sup>1*</sup>,
  <a href="https://yoonkyojung.com/">Yoonkyo Jung</a><sup>1*</sup>,
  <a href="https://haoranchen.github.io/">Haoran Chen</a><sup>1</sup>,
  <a href="https://zikuicai.github.io/">Zikui Cai</a><sup>1</sup>,
  <a href="https://aayushwhiz.github.io/">Aayush Talreja</a><sup>1</sup>,<br>
  <a href="https://sjlee.cc/">Seungjae Lee</a><sup>1</sup>,
  <a href="https://hjhyunjinkim.github.io/">Hyunjin Kim</a><sup>1</sup>,
  <a href="https://furong-huang.com/">Furong Huang</a><sup>1,2</sup><br>
  <sub><sup>*</sup>Equal contribution</sub><br>
  <sub><sup>1</sup>University of Maryland, College Park &nbsp;·&nbsp; <sup>2</sup>All Purpose AI</sub>
</p>

---

AuRA (**Au**tonomous **R**eal-to-sim **A**sset Reconstruction) builds physics-informed digital twins of deformable objects without human manipulation and without manual annotation of the recorded interaction. Three OAK-D RGB-D cameras around the workspace capture the static scene, AuRA selects a grasp that passes an execution-aware feasibility check, and a Franka Research 3 (FR3) grasps and lifts the object. The lift is recorded as synchronized multi-view RGB-D video together with the robot's joint states, so the gripper motion is measured rather than visually tracked.

> **Headline results** (paper, Sec. IV-A): across 13 deformable objects, AuRA reduces active human labor and wall-clock collection time per successfully reconstructed asset by 88.5% and 55.4% compared with human-operated collection overall, at an end-to-end reconstruction success rate of 74.3% (26/35) versus 26.9% (21/78).

**This release (v1) is the automated data collection**, from calibrating your rig to a recorded grasp-and-lift in the [PhysTwin](https://github.com/Jianghanxiao/PhysTwin) data format, plus a camera-only recorder for the human baseline. **v2 (coming soon)** adds the post-processing ([`aura/postprocessing/`](aura/postprocessing/README.md)) and the training and asset reconstruction ([`aura/reconstruction/`](aura/reconstruction/README.md)) that turn the robot-executed recordings into simulation-ready spring-mass + 3D Gaussian assets.

> [!WARNING]
> **This software moves a real robot.** Keep the emergency stop within reach, run the [pre-flight checklist](docs/usage.md#pre-flight-checklist) before every session, and read the [known limitations](#known-limitations) before the first run. Do not pass `--yes`, which skips the interactive checkpoints, until you trust your calibration and your scene.

---

## Start here

| If you want to… | Read |
|---|---|
| Set up the robot, the cameras and the workstation | [`docs/hardware.md`](docs/hardware.md), [`docs/installation.md`](docs/installation.md) |
| Get the AnyGrasp SDK, license and checkpoint | [`docs/anygrasp_license.md`](docs/anygrasp_license.md) |
| Calibrate your rig | [`docs/calibration.md`](docs/calibration.md) |
| Run a collection session end to end | [`docs/usage.md`](docs/usage.md) |
| Know what a file on disk contains | [`docs/data_format.md`](docs/data_format.md) |
| Look up the environments, versions and model weights | [`environments/README.md`](environments/README.md) |
| Look up a script's options and their defaults | `bash scripts/<script>.sh --help` |
| Fix something that broke | [`docs/troubleshooting.md`](docs/troubleshooting.md) |
| **Know what v1 does not do, and where it can surprise you** | **[Known limitations](docs/troubleshooting.md#known-limitations)** — read this before the first robot run |

---

## The pipeline in one picture

```
┌───────────────────── STAGE 1 · AUTOMATED DATA COLLECTION (v1, this release) ─────────────────────┐
│  PREPARE    01 static capture ─► 02 segment + fuse ─► 03 AnyGrasp ─► 04 feasibility filter ─┐    │
│  no robot   3× OAK-D snapshot    masks, fused cloud   6-DoF grasps   table, cameras, IK     │    │
│  motion     ┌───────────────────────────────────────────────────────────────────────────────┘    │
│             ▼                                                                                    │
│  EXECUTE    home ─► hover ─► descend ─► close gripper ─► lift ─► release + home                  │
│  FR3 moves  Enter before each step (--yes skips)         ╰─ only the lift is recorded:           │
│             │                                               3× RGB-D at 15 Hz                    │
│             ▼                                               + joint states sampled at 100 Hz     │
│  RESULT     PhysTwin-format case: color/ depth/ calibrate.pkl metadata.json + robot states       │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 ▼
┌──────────────────────── STAGE 2 · DATA POST-PROCESSING (v2, coming soon) ────────────────────────┐
│  object trajectories             controller trajectories         pre-interaction priors          │
│  segment and track the video,    recorded robot states + the     point cloud and appearance,     │
│  lifted to 3-D                   calibrated fingertip geometry   registered to the first frame   │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 ▼
┌──────────────────────── STAGE 3 · ASSET RECONSTRUCTION (v2, coming soon) ────────────────────────┐
│  PhysTwin spring-mass optimization + static 3D Gaussians bound to the mass nodes                 │
│  = a simulatable, renderable digital twin of the object                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Two things this picture does **not** show, because they matter:

1. **Every rig needs its own calibration first:** camera intrinsics, then extrinsics, then hand-eye, plus the fingertip offsets ([`docs/calibration.md`](docs/calibration.md)). The hand-eye session is recorded with the gamepad teleoperation recorder, `aura/robot/gamepad_record_franka_3oak.py`.
2. **There is a parallel human-baseline path.** `scripts/collect_hand_phystwin_one_shot.sh` records a person manipulating the object with the three cameras, in the same PhysTwin format but without robot states. This is the human-operated collection the paper compares against.

---

## Repository layout

```
AuRA/
├── README.md                  ← you are here
├── aura/                      ← the Python tools, one folder per part of the pipeline
│   ├── capture/               ← 3-camera RGB-D recorder, static snapshot (01), live alignment view
│   ├── perception/            ← Grounded-SAM-2 segmentation, multi-view fusion and filtering (02)
│   ├── grasping/              ← AnyGrasp wrapper (03), feasibility filter (04)
│   ├── robot/                 ← recording executor, standalone executor, teleop recorder
│   ├── conversion/            ← raw recordings → PhysTwin layout, calibration → calibrate.pkl
│   ├── calibration/           ← intrinsics, extrinsics, hand-eye, fingertip fit, re-anchoring
│   ├── postprocessing/        ← Stage 2, coming in v2
│   └── reconstruction/        ← Stage 3, coming in v2
├── scripts/                   ← one-shot scripts: preparation, lift (full run), human baseline
├── config/aura.env.example    ← template for config/aura.env (robot IP, camera MXIDs, paths)
├── environments/              ← conda envs AuRA + Anygrasp, robot uv project, reference exports
├── assets/                    ← FR3 home state and fingertip offsets of the authors' robot
├── calibration/               ← your rig's calibration results (shipped empty)
├── checkpoints/               ← Grounded-SAM-2 weights (not shipped)
├── data/                      ← your recorded cases (default $AURA_DATA_ROOT; git-ignored)
│   ├── franka_lift/<case>/    ← anygrasp/ (01-02), anygrasp_results*/ (03-04), phystwin/ (lift)
│   └── hand_collected/<case>/ ← one human-baseline case: PhysTwin format, no robot states
├── docs/                      ← the documentation (start with the table above)
├── third_party/fr3/           ← vendored FR3 control library, wraps pylibfranka
├── LICENSE                    ← Apache-2.0
└── THIRD_PARTY_NOTICES.md     ← third-party licenses and the code adapted from them
```

**This repository contains code and small files only.** Your configuration, calibration results, model weights, the third-party code you clone (including the AnyGrasp SDK and your license), the robot `.venv` and all recorded data stay out of git; the patterns are in [`.gitignore`](.gitignore).

---

## Installation

| You need | Details: [`docs/hardware.md`](docs/hardware.md) |
|---|---|
| **Robot** | Franka Research 3 with the Franka Hand and the Franka Control Interface (FCI), on a wired link |
| **Cameras** | 3 × Luxonis OAK-D, rigidly mounted around the workspace, each on its own USB 3 port |
| **For calibration** | a printed [ChArUco board](docs/hardware.md#charuco-board) (4 × 5 squares of 40 mm, 30 mm markers, `DICT_4X4_50`) and a PlayStation-style gamepad |
| **Workstation** | Linux x86-64, an NVIDIA GPU (driver and toolkit for CUDA 12.8), 32 GiB RAM recommended, a [PREEMPT_RT kernel](docs/hardware.md#real-time-kernel) for the robot steps, conda and [uv](https://docs.astral.sh/uv/), a [desktop session](docs/hardware.md#workstation) (stage 03's AnyGrasp debug renders and the camera previews need a display) and internet access on the first run; tested on Ubuntu 24.04 with an RTX 5090 |

> [!IMPORTANT]
> **The AnyGrasp SDK, its license and its checkpoint are not included** and may not be redistributed. Request the license first, because it takes days to arrive, but only once the rig's final network hardware is connected (for example a USB-Ethernet adapter for the robot): the license is tied to the MAC addresses of the wired network interfaces. How: [`docs/anygrasp_license.md`](docs/anygrasp_license.md).

AuRA runs in two conda envs and a uv project for the robot scripts. In short ([`docs/installation.md`](docs/installation.md) has every command):

```bash
git clone https://github.com/Yoonkyo/AuRA.git && cd AuRA
cp config/aura.env.example config/aura.env      # fill in at step 6 of docs/installation.md
conda env create -f environments/aura.yml       # env "AuRA" + POST-INSTALL STEPS in its header
conda env create -f environments/anygrasp.yml   # env "Anygrasp" + POST-INSTALL STEPS in its header
uv sync --project environments/robot            # robot env: fr3 and pylibfranka
```

The POST-INSTALL STEPS add what a conda file cannot express, including the Grounded-SAM-2 build and your AnyGrasp SDK, license and checkpoint. Then download the [Grounded-SAM-2 weights](docs/installation.md#3-grounded-sam-2-and-model-weights) and [calibrate your rig](docs/calibration.md).

---

## Quick start

Run the [pre-flight checklist](docs/usage.md#pre-flight-checklist), place the object on the table inside the calibrated workspace, with the arm out of the cameras' view (for example at home), then prepare the case from the repository root:

```bash
conda activate AuRA; set -a; source config/aura.env; set +a
CASE=green_toy_1                                  # segmentation prompt from the name: "green toy."
bash scripts/anygrasp_poses_one_shot.sh "$CASE"   # 1. prepare: stages 01-04, no robot motion
```

**Inspect before anything moves**, and leave the object and the cameras untouched until the run: the grasp is computed from this snapshot.

- In `$AURA_DATA_ROOT/franka_lift/$CASE/`, read the filter report `anygrasp_results/grasps_table_ik_report.json` and look at the renders under `anygrasp/` and `anygrasp_results*/` ([inspect the grasps](docs/usage.md#inspect-the-grasps)).
- Do the [dry run and the hover-only test](docs/usage.md#dry-run-and-hover-only-test), from a shell without an active conda env.

Then execute and record. Without `--yes`, the run waits for Enter before each robot step:

```bash
conda activate AuRA                                     # the dry run and hover test ran without it
bash scripts/run_anygrasp_to_lift_one_shot.sh "$CASE"   # 2. execute and record (the lift one-shot)
```

**Verify the recording.** An aborted lift still writes a normal-looking recording, so first check the console for `[robot] ERROR`. Then read `phystwin/recording_summary.json` (saved and dropped timesteps) and watch the videos `phystwin/color/0.mp4` to `2.mp4` ([details](docs/usage.md#verify-the-recording)).

**Human baseline** (cameras only, no robot): `bash scripts/collect_hand_phystwin_one_shot.sh colorful_rope_1`. Once `[READY] Recording is active` appears, a person manipulates the object; press Enter once to stop (never Ctrl-C, which loses the take), then choose the frames to keep ([details](docs/usage.md#human-baseline-collection)).

Cases go to `$AURA_DATA_ROOT` (`data/` by default, see the [layout](#repository-layout)); [`docs/data_format.md`](docs/data_format.md) describes every file. Cases and logs record your absolute paths, camera serial numbers and robot address, so read [Before sharing data](docs/data_format.md#before-sharing-data) before you publish recordings.

---

## Known limitations

v1 keeps the behaviour and the default parameters of the authors' pipeline and documents its rough edges instead of changing them. Know these before the first run; [`docs/troubleshooting.md`](docs/troubleshooting.md#known-limitations) has the full list, with workarounds.

- **The feasibility filter is not a full collision checker.** It checks fingertip-table clearance, camera keep-out and IK along the path; the arm links, the hand body, self-collision, the object and the camera mounts are not modelled. Judge the scene and the renders yourself.
- **Grasp success is effectively not checked.** With the default gripper settings the lift goes ahead even when the gripper closed on nothing. Watch the checkpoints.
- **The teleoperation recorder homes the arm on launch**, without a prompt; `--home-on-start False` skips it.
- **Safe index 0 is the accepted grasp with the least joint motion** from home to hover, not necessarily AnyGrasp's top grasp shown in `debug_best_grasp.png`; `--safe-index N` executes another one.
- **The paper's cases used stricter filter thresholds:** `--clearance-m 0.005 --robust-joint-margin-rad 0.2` instead of the preparation defaults 0.003 m and 0.16 rad. On a prepared case the full run re-runs only the filter, so pass them to it as well, after `--` ([Step 1](docs/usage.md#step-1-preparation-no-robot-motion), [Step 2](docs/usage.md#step-2-full-autonomous-run)).

---

## Citation

If you use AuRA, please cite the paper. The arXiv identifier will be added once it is public.

```bibtex
@article{yu2026aura,
  title  = {{AuRA}: From Atoms to Bits through Autonomous Robot Interaction},
  author = {Yu, Xinchen and Jung, Yoonkyo and Chen, Haoran and Cai, Zikui and
            Talreja, Aayush and Lee, Seungjae and Kim, Hyunjin and Huang, Furong},
  journal={arXiv preprint arXiv:2610.34452},
  year={2026}
}
```

## License

- **AuRA's own code:** [Apache-2.0](LICENSE), Copyright 2026 The AuRA Authors. Third-party components, and the AuRA files that adapt code from them: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- **Not included, under their own terms:** the AnyGrasp SDK, license and checkpoint; Grounded-SAM-2 and its weights, which you clone and download during installation.
- **Vendored `third_party/fr3`:** its license, which also covers the parts of the teleoperation recorder adapted from it, is still to be confirmed by its authors.

## Acknowledgements

To be added. AuRA builds on AnyGrasp, Grounded-SAM-2 (SAM 2 and Grounding DINO), PyRoki, PhysTwin, fr3, pylibfranka and DepthAI.
