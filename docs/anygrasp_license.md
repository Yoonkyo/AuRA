# AnyGrasp SDK, license and checkpoint

AuRA uses the `grasp_detection` module of the [AnyGrasp SDK](https://github.com/graspnet/anygrasp_sdk) in stage 03 of `scripts/anygrasp_poses_one_shot.sh`, to propose 6-DoF grasps on the fused object point cloud. **The SDK, its license files and its checkpoint are not part of this repository and may not be redistributed.** You obtain them from the AnyGrasp authors under their terms.

The pipeline finds all three through `AURA_ANYGRASP_SDK` in `config/aura.env`. Its default, `third_party/anygrasp_sdk`, is git-ignored. If `config/aura.env` does not exist yet, create it from the repository root with `cp config/aura.env.example config/aura.env` ([installation step 1](installation.md#1-clone-and-create-your-configuration)). The template's default paths are enough for the steps below.

> [!IMPORTANT]
> Request the license first. It is locked to one machine and takes days to arrive.

## What you need

| Item | Location | Source |
|---|---|---|
| AnyGrasp SDK | `$AURA_ANYGRASP_SDK` | public git repository |
| License: `licenseCfg.json`, `<name>.lic`, `<name>.public_key`, `<name>.signature` | `$AURA_ANYGRASP_SDK/grasp_detection/license/` | the AnyGrasp authors, for one machine |
| Detection checkpoint `checkpoint_detection.tar` (296,408,957 bytes) | `$AURA_ANYGRASP_SDK/grasp_detection/checkpoint_detection.tar` | the AnyGrasp authors |
| conda env `Anygrasp` | your conda installation | [`environments/anygrasp.yml`](../environments/anygrasp.yml) |
| `ifconfig`, `iwconfig` | on `PATH` | `sudo apt install net-tools wireless-tools` |

AuRA does not use the SDK's grasp-tracking module or its `checkpoint_tracking.tar`.

## 1. Create the `Anygrasp` environment

```bash
conda env create -f environments/anygrasp.yml
```

conda may ask you once to accept the Anaconda Terms of Service for the `defaults` channel. Accept at the prompt, or run the `conda tos accept` commands it prints and start `conda env create` again.

AnyGrasp needs an environment of its own:

- its compiled binaries need `numpy` < 1.24, while the `AuRA` env uses numpy 1.26;
- its license library and `license_checker` link OpenSSL 1.1 (`libcrypto.so.1.1`), which Ubuntu 24.04 no longer ships. The env pins `python=3.10.0` and `openssl=1.1.1w` to provide it.

The header of `environments/anygrasp.yml` lists **POST-INSTALL STEPS**: `graspnetAPI`, MinkowskiEngine for CUDA 12.8, the SDK with `pointnet2`, the binaries, the license and the checkpoint. Only the SDK clone and the license request are needed first; the builds can wait.

## 2. Clone the SDK

To keep the SDK somewhere other than the default `third_party/anygrasp_sdk`, set `AURA_ANYGRASP_SDK` in `config/aura.env` before you run this block. If this shell has loaded the file before, run the block in a new terminal or after `unset AURA_ANYGRASP_SDK`.

```bash
conda activate Anygrasp
set -a; source config/aura.env; set +a
git clone https://github.com/graspnet/anygrasp_sdk.git "$AURA_ANYGRASP_SDK"
git -C "$AURA_ANYGRASP_SDK" checkout 3cb4a593e1a7cdde6ac897cfce49c26e1c41db9c   # the commit used for the paper
```

Keep the SDK folder out of any repository you publish: it will contain your license. The stage-03 and 03b logs in `<case>/logs/` also give the path of your license folder and the names of your license files ([Before sharing data](data_format.md#before-sharing-data)).

## 3. Request the license

The license is **machine-locked** and not transferable; the authors' license does not work on your machine. `license_checker` computes a feature id from the MAC addresses of the active wired network interfaces. It skips wireless, `docker*`, `br-*` and `veth*` interfaces. Therefore:

- connect and bring up the rig's final network hardware first, including any USB-Ethernet adapter for the robot network;
- after the license is issued, adding or removing a wired interface can change the id and invalidate the license.

Print the feature id of this machine:

```bash
(cd "$AURA_ANYGRASP_SDK/license_registration" && LD_LIBRARY_PATH="$CONDA_PREFIX/lib" ./license_checker -f)
```

`LD_LIBRARY_PATH` lets the checker find the env's `libcrypto.so.1.1`. The id is printed without a final newline: copy only the id, not the prompt (or zsh's `%`) that follows it. The checker exits with status 1 even when it succeeds. Submit the id through the registration form linked from the SDK README (<https://forms.gle/XVV3Eip8njTYJEBo6> at the time of writing). A reply usually takes a few working days; check your spam folder if nothing arrives.

Meanwhile, finish the other post-install steps of `environments/anygrasp.yml`: 1 (`graspnetAPI`), 2 (MinkowskiEngine), the `pointnet2` build of step 3 (the SDK is already cloned) and 4 (the binaries).

## 4. Install and verify the license

If you continue in a new terminal, first run `conda activate Anygrasp` and `set -a; source config/aura.env; set +a` from the repository root, as in step 2. The reply contains a `.zip` with a `license/` folder. Unzip it into `grasp_detection/`, for example with `unzip <file>.zip -d "$AURA_ANYGRASP_SDK/grasp_detection"`, so that the files end up here (not in `license/license/`):

```
$AURA_ANYGRASP_SDK/grasp_detection/license/
├── licenseCfg.json
├── <name>.lic
├── <name>.public_key
└── <name>.signature
```

Then check it:

```bash
(cd "$AURA_ANYGRASP_SDK/license_registration" && LD_LIBRARY_PATH="$CONDA_PREFIX/lib" ./license_checker -c ../grasp_detection/license/licenseCfg.json)
```

## 5. Checkpoint

The detection checkpoint `checkpoint_detection.tar` is licensed and has no public download link; the SDK's `grasp_detection/README.md` only says to put the model weights under `log/`. Obtain it from the AnyGrasp authors together with your license, and place it at `$AURA_ANYGRASP_SDK/grasp_detection/checkpoint_detection.tar`.

- The preparation script looks there first, then anywhere below `$AURA_ANYGRASP_SDK`.
- `--anygrasp-checkpoint PATH` overrides both locations, but only if PATH is an existing file. A PATH that does not exist is silently replaced by the first `checkpoint_detection.tar` found below `$AURA_ANYGRASP_SDK`, and neither `<case>/logs/03_anygrasp_pose_generation.log` nor `<case>/one_shot_summary.json` records which file was loaded. Check PATH with `ls -l` before the run, and make it absolute: stage 03 resolves a relative PATH against its working directory, `$AURA_ANYGRASP_SDK/grasp_detection`, but checks it earlier from another directory, usually the repository root.
- The copy used for the paper has sha256 `a05c3690b95c8b65e78b1bb8a28f1d5ca96613391946e450afacae840bbcf7b2`; compare with `sha256sum "$AURA_ANYGRASP_SDK/grasp_detection/checkpoint_detection.tar"`.

## 6. Binaries for your Python version

`grasp_detection/gsnet.so` and `grasp_detection/lib_cxx.so` must match the Python version of the env: `cpython-310` for `Anygrasp`. Post-install step 4 copies them from `grasp_detection/gsnet_versions/` and `license_registration/lib_cxx_versions/`. `scripts/anygrasp_poses_one_shot.sh` copies them too, but only when they are missing. If you ever change the env's Python minor version, delete both files so that the right ones are copied.

## 7. Check and run

The sanity check from the header of `environments/anygrasp.yml`:

```bash
conda activate Anygrasp
set -a; source config/aura.env; set +a
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$AURA_ANYGRASP_SDK/grasp_detection${PYTHONPATH:+:$PYTHONPATH}"
cd "$AURA_ANYGRASP_SDK/grasp_detection"
python -c "import numpy, torch, MinkowskiEngine, pointnet2, open3d; from gsnet import AnyGrasp; assert numpy.__version__ < '1.24'; print('Anygrasp env OK')"
```

The check passes only if it prints `Anygrasp env OK`. Without a valid license, `from gsnet import AnyGrasp` prints `[ERROR] license failed to pass!` instead, and the command still exits with status 0. Before your license arrives, leave out `from gsnet import AnyGrasp;` to check only the builds.

The two `export`s stay in the shell, also after `conda deactivate`, so start robot scripts and the lift one-shot from a new terminal ([hardware.md, Shell](hardware.md#real-time-kernel)).

Stage 03 of the preparation script sets up the same runtime by itself:

- the `Anygrasp` env;
- `$CONDA_PREFIX/lib` on `LD_LIBRARY_PATH`, for `libcrypto.so.1.1` (the license library `lib_cxx.so` has no RPATH);
- `$AURA_ANYGRASP_SDK/grasp_detection` on `PYTHONPATH`, for `from gsnet import AnyGrasp`;
- **the working directory `$AURA_ANYGRASP_SDK/grasp_detection`**. The license is looked up relative to it, and the license library writes a `.TimeRecord` file there, so the folder must be writable.

To run AnyGrasp by hand on a case, continue in the same shell, inside `grasp_detection/`:

```bash
python "$AURA_REPO_ROOT/aura/grasping/anygrasp_poses_from_pointcloud.py" \
  --input "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp/observation/observation_frame000_filtered.npz" \
  --checkpoint_path "$AURA_ANYGRASP_SDK/grasp_detection/checkpoint_detection.tar" \
  --output_dir "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp_results" \
  --top_k 20 --max_gripper_width 0.08 --top_down_grasp --save_debug_images
```

- `--max_gripper_width 0.08` (the SDK's default is 0.1) matches the Franka Hand's stroke.
- `--no_collision_detection` reproduces the preparation script's automatic retry (stage 03b) only together with `--output_dir "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp_results_no_collision"`. Never write a no-collision run into `anygrasp_results/`: no output file records the setting, and the preparation script uses that folder's `grasps.json` first, as the collision-checked result.
- `--save_debug_images` renders through a hidden Open3D window and therefore needs a display.
- `--debug` writes the same images, so it needs a display too, and then opens two blocking Open3D windows, one after the other.

The output files are described in [data_format.md](data_format.md#anygrasp-output).

## Moving to another machine

A new workstation needs a new license: run `license_checker -f` there and request again. The SDK clone and the checkpoint can be reused. Run `license_checker -f` again after any change of the wired network interfaces, and request a new license if the id changed.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `libcrypto.so.1.1: cannot open shared object file` | The `Anygrasp` env is not active, the env lacks OpenSSL 1.1, or `$CONDA_PREFIX/lib` is not on `LD_LIBRARY_PATH`. Run `conda activate Anygrasp`, then the checker as shown above. If the error stays, recreate the env from `environments/anygrasp.yml`, which pins `python=3.10.0` and `openssl=1.1.1w`. |
| `license_checker -f` prints no id (for example `ifconfig: not found`), or AnyGrasp reports `feature id doesn't match the hardware` | `ifconfig` or `iwconfig` is missing: install them with `sudo apt install net-tools wireless-tools`. Or the wired network interfaces changed since the license was issued: restore them as they were (reconnect and bring up a removed adapter, such as a USB-Ethernet adapter, or remove or bring down an added one), or request a new license for the new id ([Moving to another machine](#moving-to-another-machine)). |
| AnyGrasp prints `license failed to pass!` although `license_checker -c` succeeds | AnyGrasp was not started from `$AURA_ANYGRASP_SDK/grasp_detection`, or that folder is not writable. Start it from that folder, as stage 03 does ([step 7](#7-check-and-run)), and make the folder writable. |
| `AnyGrasp SDK not found at ...` | Set `AURA_ANYGRASP_SDK` in `config/aura.env` to your clone; if your shell has loaded the file before, open a new terminal or run `unset AURA_ANYGRASP_SDK` before you retry. |
| `No gsnet binary for cpython-3XY` | The SDK has no binary for the Python of the active env; run stage 03 in `Anygrasp` (Python 3.10). |
| `Cannot find checkpoint_detection.tar` | Place the checkpoint as in step 5, or pass `--anygrasp-checkpoint` with the absolute path of an existing file. |

More problems and their fixes are in [troubleshooting.md](troubleshooting.md).

## Terms and citation

The AnyGrasp SDK, license and checkpoint are governed by the AnyGrasp authors' terms, not by this repository's Apache-2.0 license. If you use AnyGrasp through AuRA, cite the AnyGrasp papers as listed in the SDK README.
