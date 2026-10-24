# `environments/` — software environments

AuRA v1 runs in three separate environments, each isolated for an ABI reason:
the AnyGrasp binaries need numpy < 1.24 and OpenSSL 1.1, the main pipeline needs
numpy 1.26, and the robot scripts need CPython 3.11/3.12 with `pylibfranka` on a
real-time kernel.
The scripts in [`../scripts/`](../scripts/) switch between them per stage. Run
robot scripts by hand from a shell without an active conda env, so that no conda
`LD_LIBRARY_PATH` leaks into the real-time process ([`docs/hardware.md`, Shell](../docs/hardware.md#real-time-kernel)).

| File | What it is |
|---|---|
| [`aura.yml`](aura.yml) | conda env **`AuRA`** (Python 3.10): capture, calibration, conversion, segmentation, point-cloud fusion, PyRoki filtering. Curated minimal set; the post-install steps are in its header. |
| [`anygrasp.yml`](anygrasp.yml) | conda env **`Anygrasp`** (Python 3.10.0 with OpenSSL 1.1, numpy 1.23.5): AnyGrasp inference only. MinkowskiEngine, SDK, license and checkpoint steps are in its header. |
| [`robot/`](robot/) | uv project (`pyproject.toml` + `uv.lock`) for every script that imports `fr3` / `pylibfranka`. |
| [`reference/AuRA_environment.yml`](reference/AuRA_environment.yml), [`reference/AuRA_pip_freeze.txt`](reference/AuRA_pip_freeze.txt) | Sanitized full export of the authors' `AuRA` env (2026-09-17). Archival, see [below](#reference-exports). |
| [`reference/Anygrasp_environment.yml`](reference/Anygrasp_environment.yml), [`reference/Anygrasp_pip_freeze.txt`](reference/Anygrasp_pip_freeze.txt) | The same for `Anygrasp`. |

The scripts find the environments through `config/aura.env`: `AURA_ENV=AuRA`,
`ANYGRASP_ENV=Anygrasp`, `AURA_ROBOT_PROJECT=<repo>/environments/robot`. If you
create the conda envs under other names (`conda env create -n ...`), change them
there.

## Which environment runs which stage

| Stage | Script(s) | Environment |
|---|---|---|
| 3-OAK capture, manual alignment, static AnyGrasp frame (stage 01) | `aura/capture/record_3oak.py`, `manual_align_3oak.py`, `anygrasp_data_collection.py` | `AuRA` |
| Calibration: intrinsics, extrinsics, hand-eye, fingertip, 3-camera anchor | `aura/calibration/*.py` | `AuRA` |
| Raw recording → PhysTwin-style layout | `aura/conversion/*.py` | `AuRA` |
| Segmentation + observation point cloud (stage 02) | `aura/perception/*.py` | `AuRA` + Grounded-SAM-2 clone and weights, GPU |
| AnyGrasp inference (stage 03) | `aura/grasping/anygrasp_poses_from_pointcloud.py` | `Anygrasp` + your SDK, license and checkpoint |
| Table-collision + PyRoki IK/FK filter (stage 04) | `aura/grasping/filter_grasps_table_collision_pyroki.py` | `AuRA` |
| Grasp execution + lift recording | `aura/robot/execute_anygrasp_franka_record_lift.py` (imports its sibling `execute_anygrasp_franka.py`) | `robot` uv project, RT kernel |
| Gamepad teleoperation recording | `aura/robot/gamepad_record_franka_3oak.py` | `robot` uv project, RT kernel |

- `scripts/anygrasp_poses_one_shot.sh` (stages 01–04) runs `conda activate "$AURA_ENV"` or `"$ANYGRASP_ENV"` per stage and sets up the AnyGrasp runtime (`LD_LIBRARY_PATH`, `PYTHONPATH`, working directory; see the RUNTIME block of `anygrasp.yml`).
- `scripts/run_anygrasp_to_lift_one_shot.sh` calls that preparation script, runs the robot step with `uv run --project "$AURA_ROBOT_PROJECT"` (override: `--uv-project`) and the conversions with `conda run -n "$AURA_ENV"`.
- `scripts/collect_hand_phystwin_one_shot.sh` runs every step with the python of `$AURA_ENV` (override with `--aura-python` or `AURA_PYTHON`).

## Installation

Follow [`docs/installation.md`](../docs/installation.md), which walks through the
installation step by step. As it says, request the AnyGrasp license first: the
license is machine-locked and takes days to arrive. The sections below are the
reference that its steps link to.

### System requirements

- Linux x86_64. Reference machine: Ubuntu 24.04 (gcc/g++ 13.3, glibc 2.39). depthai 3.6.1 needs glibc ≥ 2.28, `pylibfranka` 0.20.5 needs glibc ≥ 2.34.
- An NVIDIA GPU with a driver for the CUDA 12.8 runtime (≥ 570). Reference: RTX 5090 (Blackwell, sm_120). The pinned `torch 2.11.0+cu128` wheels contain kernels from sm_75 to sm_120; older `cu121`/`cu124` builds have no Blackwell kernels.
- A CUDA 12.8 toolkit with `nvcc`, found through `CUDA_HOME` (default `/usr/local/cuda`; add `$CUDA_HOME/bin` to `PATH`), to build `groundingdino._C` (`AuRA`) and MinkowskiEngine + pointnet2 (`Anygrasp`). The toolkit is needed only for these builds: at runtime the compiled extensions use the `libcudart.so.12` and `libcusparse.so.12` that `import torch` has already loaded from torch's `nvidia-*-cu12` wheels.
- `build-essential`, `ninja-build`, `git`, and network access for pip, the git clones and the [first-run downloads](#first-run-downloads). `conda env create` already needs `build-essential`: it compiles `grasp-nms` (`Anygrasp`) and `pyliblzfse` (`AuRA`) from source. Without `ninja-build`, the MinkowskiEngine build falls back to a slower backend and takes about 7 minutes instead of about 1.
- For the AnyGrasp license: `ifconfig` and `iwconfig` on `PATH` (`sudo apt install net-tools wireless-tools`; Ubuntu 24.04 does not install `net-tools` by default). The license library runs them to compute the machine feature id from the MAC address(es) of the active wired interfaces (wireless, `docker*`, `br-*` and `veth*` interfaces are skipped). Without them `license_checker -f` fails and AnyGrasp rejects the license (`feature id doesn't match the hardware`). Adding or removing a wired NIC, e.g. a USB-Ethernet adapter for the FR3 network, can change the id and invalidate the license, so connect the rig's network hardware before you run `license_checker -f`.
- Three Luxonis OAK-D cameras on USB3. Linux needs the Movidius udev rule (present on the reference machine):
  ```bash
  echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```
- Disk: ~1.6 GB of Grounded-SAM-2 weights, ~440 MB for `bert-base-uncased`, 283 MiB for the AnyGrasp checkpoint, and about 9 GB per conda env.
- Building MinkowskiEngine with CUDA 12.8 needs a sudo edit of the libstdc++ system header `bits/shared_ptr_base.h` (the AnyGrasp SDK README's fix; without it the build fails with gcc 13 on Ubuntu 24.04). Step 2 of `anygrasp.yml` makes the edit, keeps a `.orig` backup and restores the stock header right after that build.

## Robot environment (`robot/`)

```bash
uv sync --project environments/robot                        # creates environments/robot/.venv
uv run --project environments/robot python aura/robot/gamepad_record_franka_3oak.py --help
```

- **Python 3.11 or 3.12 only** (`requires-python = ">=3.11,<3.13"`): `fr3` requires ≥ 3.11, and `pylibfranka` 0.20.5 publishes no cp313 wheel (its wheels cover cp39 to cp312; the cp311/cp312 ones are `manylinux_2_34_x86_64`). uv prefers interpreters it manages over the system Python and downloads one if none matches, so a plain `uv sync` can pick a uv-managed 3.11, which gets other numpy and scipy versions from the lock (2.4.6 / 1.17.1 instead of 2.5.4 / 1.18.1). The authors used Ubuntu 24.04's CPython 3.12: `uv sync --project environments/robot --python 3.12` selects 3.12.
- `pyproject.toml` lists every runtime dependency: `fr3` (editable from [`../third_party/fr3`](../third_party/fr3/)), `pylibfranka==0.20.5`, `depthai==3.6.1`, `opencv-python==4.13.0.92`, `pygame==2.6.1`, numpy and scipy; `uv.lock` pins the rest. `uv sync` (which removes packages that are not in the lock) is therefore safe.
- **Real-time kernel.** `pylibfranka.Robot` is created with its default `RealtimeConfig.kEnforce`, so connecting to the arm (`FR3()` or `Robot(ip)`) fails with a real-time error without a PREEMPT_RT kernel and real-time scheduling rights. Only the robot steps need this. The reference machine runs Ubuntu's `linux-realtime` PREEMPT_RT kernel (Ubuntu Pro); setup and checks are in [`docs/hardware.md`](../docs/hardware.md#real-time-kernel). See also [`../third_party/fr3/README.md`](../third_party/fr3/README.md).

## Model weights

Grounded-SAM-2 (stage 02) needs three files in `$AURA_GSAM_CHECKPOINTS` (default `checkpoints/groundedSAM/`):

| File | Source | Bytes | sha256 |
|---|---|---|---|
| `sam2.1_hiera_large.pt` | <https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt> | 898,083,611 | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` |
| `groundingdino_swint_ogc.pth` | <https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth> | 693,997,677 | `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799` |
| `GroundingDINO_SwinT_OGC.py` | copy from your clone: `$AURA_GSAM_REPO/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py` ([upstream at the pinned commit](https://raw.githubusercontent.com/IDEA-Research/Grounded-SAM-2/b7a9c29f196edff0eb54dbe14588d7ae5e3dde28/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py)) | 1,006 | `172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1` |

The config copy the authors used lacked the final newline (1,005 bytes, sha256
`7010c6c7026e1674c8c2a7a1abf6e86dd104bf0fde4d63b3b378f35dda8aa189`) and is otherwise identical.

[Step 3 of `docs/installation.md`](../docs/installation.md#3-grounded-sam-2-and-model-weights)
downloads the two weight files and copies the config file from your clone. To
check the weights, from the repository root:

```bash
set -a; source config/aura.env; set +a
(cd "$AURA_GSAM_CHECKPOINTS" && sha256sum -c - <<'EOF'
2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318  sam2.1_hiera_large.pt
3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799  groundingdino_swint_ogc.pth
EOF
)
```

**AnyGrasp** (stage 03): `checkpoint_detection.tar` (296,408,957 bytes, 283 MiB) is
licensed and not redistributable. Obtain it from the AnyGrasp authors together
with your license (see the [AnyGrasp SDK](https://github.com/graspnet/anygrasp_sdk))
and place it at `$AURA_ANYGRASP_SDK/grasp_detection/checkpoint_detection.tar`.
The copy used for the paper has sha256
`a05c3690b95c8b65e78b1bb8a28f1d5ca96613391946e450afacae840bbcf7b2`.
`checkpoint_tracking.tar` is not used by v1.

## First-run downloads

Two more downloads happen automatically the first time they are needed. Network
is needed once; on an offline robot machine, pre-populate the caches.

| What | Triggered by | Cached in |
|---|---|---|
| [`bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) (~440 MB; the paper used revision `86b5e0934494bd15c9632b12f734a8a67f723594`) | GroundingDINO's text encoder, the first time stage 02 runs | `~/.cache/huggingface/hub/` |
| [`franka_description`](https://github.com/frankarobotics/franka_description) at commit `72baf5bf4e88eaec27f0eb61be1b20a001abf2ab` (pinned by `robot_descriptions` 3.1.0) | `load_robot_description("fr3_description")`, the first time the PyRoki filter (stage 04) runs | `~/.cache/robot_descriptions/` |

To fetch both ahead of time, inside the `AuRA` env:

```bash
python -c "from transformers import AutoTokenizer, BertModel; AutoTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"
python -c "from robot_descriptions.loaders.yourdfpy import load_robot_description; load_robot_description('fr3_description')"
```

After that, `HF_HUB_OFFLINE=1` stops transformers from contacting the hub.

## The opencv-contrib 4.7.0 hazard

The `AuRA` env must contain exactly one OpenCV: `opencv-contrib-python==4.7.0.72`.
The ChArUco calibration tools use both of OpenCV's ArUco APIs, and this pinned
version (the authors' reference version) provides both:

- `aura/calibration/Extract_intrinsic.py` detects and calibrates with the legacy contrib functions `cv2.aruco.detectMarkers`, `interpolateCornersCharuco` and `calibrateCameraCharucoExtended`. On 4.7.0 it builds the board and the detector parameters with the new `CharucoBoard(...)` and `DetectorParameters()` constructors. Its `CharucoBoard_create` / `DetectorParameters_create` branches are fallbacks for OpenCV < 4.7 only; 4.7.0 has neither function.
- `aura/calibration/Extract_extrinsic.py`, `robot_world_calib_charuco.py` and `calibrate_3oak_anchor108.py` use the newer `cv2.aruco.CharucoDetector` / `CharucoBoard` API.

`opencv-python`, `opencv-python-headless` and `opencv-contrib-python` all install
into the same `cv2/` directory. Any later `pip install` that pulls in another
OpenCV wheel silently replaces the `cv2` the calibration needs. Of the packages
`aura.yml` installs, only supervision declares an OpenCV wheel (`opencv-python`),
so `aura.yml` installs supervision with `--no-deps`. sam2 and groundingdino are
also installed with `--no-deps`, but neither pulls in OpenCV: at the pinned
Grounded-SAM-2 commit, groundingdino declares no dependencies, and sam2 lists
`opencv-python` only in its `notebooks`/`dev` extras. The v2-only `rembg` listed
in `reference/` declares `opencv-python-headless`, so installing it into `AuRA`
would replace `cv2` as well. Check after every pip operation in the env:

```bash
python -c "import cv2; assert cv2.__version__ == '4.7.0' and hasattr(cv2.aruco, 'interpolateCornersCharuco'), cv2.__version__; print('cv2 OK')"
```

If the check fails, remove every OpenCV wheel and reinstall the pinned one:

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install --no-deps opencv-contrib-python==4.7.0.72
```

Keep a GUI build (not `-headless`): `aura/capture/record_3oak.py --preview True`
opens an OpenCV window. The other environments are not affected: `Anygrasp` has
`opencv-python==4.10.0.84` (graspnetAPI imports `cv2`), and the robot venv has
`opencv-python==4.13.0.92`.

## Reference exports

`reference/` holds the `conda env export --no-builds` and `pip freeze` captures
of the authors' two conda environments (2026-09-17), sanitized for release:

- the `prefix:` line (a local install path) is removed;
- pip entries that pip reported with a local build path are commented out: conda-built packages (already pinned in the conda section) and the v2-only PhysTwin CUDA builds;
- editable and locally built packages (`groundingdino`, `sam-2`, `pyroki`, `pytorch3d`, `MinkowskiEngine`, `pointnet2`) are commented out with how to install each; `jaxls` and `utils3d` become pinned git URLs;
- `--extra-index-url https://download.pytorch.org/whl/cu128` is added for the `+cu128` torch builds, and the deprecated `sklearn` shim is commented out;
- inert or inconsistent entries are annotated, never changed: the conda `opencv=4.13.0` and the CUDA libraries left by `pytorch-cuda=12.1` in `AuRA`, and `opencv-python 4.13.0.92` / `scikit-image 0.25.2` in `Anygrasp`, whose metadata does not accept `numpy==1.23.5`. `openssl=1.1.1w` in `Anygrasp` is marked as required, not inert.

These files also list packages that only v2 (PhysTwin, TRELLIS) needs. They are
for exact comparison only and untested as `conda env create` (the `Anygrasp`
one cannot resolve as is). Install from `aura.yml` and `anygrasp.yml`.

## Notes

- **Packages in your user site override both conda envs.** `~/.local/lib/python3.10/site-packages` is on the path of both envs (Python 3.10; unlike a venv, a conda env does not hide it), ahead of the env's own packages. A numpy, OpenCV or other package installed there with `pip install --user` (on Ubuntu 22.04 also by a `pip3 install` without sudo) overrides the pins in every stage, and the cv2 repair above cannot remove a shadowing numpy. Check with `python -m site` in each env. If `USER_SITE` exists and holds such packages, remove them, or `export PYTHONNOUSERSITE=1` before every run and also before `conda env create` (pip in the env counts them as installed; recreate an env built without it). The robot venv ignores the user site.
- Segmentation needs no `PYTHONPATH`: `aura/perception/aura_segment_static_target_image.py` puts `$AURA_GSAM_REPO` at the front of `sys.path` before importing groundingdino/sam2, so `AURA_GSAM_REPO` in `config/aura.env` must point at your Grounded-SAM-2 clone (default `third_party/Grounded-SAM-2`).
- The curated files pin every package the v1 scripts import plus the dependencies that matter for compatibility; pip resolves the remaining transitive dependencies (e.g. of open3d and matplotlib) at install time. `reference/` records the exact versions used for the paper.
- `safetensors==0.8.0rc1` is a pre-release, kept as captured.
- In `Anygrasp`, `opencv-python 4.10.0.84` and `scikit-image 0.24.0` replace the reference env's 4.13.0.92 / 0.25.2, which require a newer numpy than 1.23.5. The v1 AnyGrasp script imports neither directly.
- `pip check` in `AuRA` prints `depthai 3.6.1 is not supported on this platform`. This comes from depthai's multi-tag wheel name and is harmless. An env built from `aura.yml` also prints `supervision 0.28.0 requires opencv-python, which is not installed.` This is expected too: supervision is installed `--no-deps`, and `opencv-contrib-python` provides `cv2`. Do **not** install `opencv-python` to silence it; that replaces the `cv2` the calibration needs (see [the opencv-contrib 4.7.0 hazard](#the-opencv-contrib-470-hazard)). The authors' env does not print this line only because its conda `opencv` package left `opencv-python` metadata stubs behind.
- `pip check` in `Anygrasp` prints `graspnetapi 1.2.10 requires sklearn, which is not installed.` This is expected: graspnetAPI declares the deprecated `sklearn` PyPI shim, so step 1 of `anygrasp.yml` installs graspnetAPI with `--no-deps`, and the pinned `scikit-learn` provides the `sklearn` module. Do not install the `sklearn` shim to silence it.
- `Anygrasp` pins `python=3.10.0` and `openssl=1.1.1w` on purpose. The AnyGrasp license library (`grasp_detection/lib_cxx.so`) and `license_checker` link `libcrypto.so.1.1` and have no RPATH. Current Python 3.10.x builds come with OpenSSL 3, and `from gsnet import AnyGrasp` then fails because `lib_cxx.so` cannot load (`libcrypto.so.1.1: cannot open shared object file`). Stage 03 puts `$CONDA_PREFIX/lib` on `LD_LIBRARY_PATH` (RUNTIME block of `anygrasp.yml`); run `license_checker` the same way (step 5).
- The AnyGrasp license library writes a `.TimeRecord` file into its working directory (`$AURA_ANYGRASP_SDK/grasp_detection/`), so that directory must be writable.
