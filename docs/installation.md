# Installation

AuRA v1 runs in three software environments: the conda env `AuRA` (capture, calibration, perception, grasp filter), the conda env `Anygrasp` (AnyGrasp inference) and a [uv](https://docs.astral.sh/uv/) project for the robot scripts. This page lists the requirements and walks through the installation in six steps. Versions, the reasons for the pins and the model weights are in [`environments/README.md`](../environments/README.md); the hardware is described in [hardware.md](hardware.md).

## Requirements

Set up the robot, the cameras, the ChArUco board, the gamepad and the Linux x86-64 workstation as described in [hardware.md](hardware.md): GPU, RAM, disk, a desktop session (stage 03 and the camera previews need a display) and, for the robot steps, a PREEMPT_RT kernel with real-time rights for your user. A session runs the GPU stages under the same kernel, so the NVIDIA driver must load under it too ([hardware.md](hardware.md#real-time-kernel)). Before step 1, install (details in [System requirements](../environments/README.md#system-requirements)):

- an NVIDIA driver for CUDA 12.8 (>= 570) and a CUDA 12.8 toolkit with `nvcc` (for the `groundingdino._C`, MinkowskiEngine and `pointnet2` builds);
- conda and [uv](https://docs.astral.sh/uv/);
- `build-essential`, `git`, `ninja-build` (recommended), and `net-tools` and `wireless-tools` (for the AnyGrasp license tool).

You also need internet access during the installation and the first run, and your own AnyGrasp license and checkpoint ([step 4](#4-anygrasp-sdk-license-and-checkpoint)). The Movidius udev rule for the cameras is part of [step 6](#6-configure-configauraenv).

If `~/.local/lib/python3.10/site-packages` exists and holds packages (from `pip install --user`, or on Ubuntu 22.04 from a `pip3 install` without sudo), deal with them before any `conda env create`: both conda envs (Python 3.10) import them ahead of their own packages, so they override the pins, and pip in the env counts them as installed. Remove them, or `export PYTHONNOUSERSITE=1` before `conda env create`, the later install steps and every run; recreate an env that you created before doing so ([user-site note](../environments/README.md#notes)).

## Installation steps

All commands run from the repository root. Once the rig's final network hardware is connected (the AnyGrasp license is locked to the MAC addresses of the workstation's wired network interfaces), start with the license, because it takes days to arrive: run [step 1](#1-clone-and-create-your-configuration) (its `cp config/aura.env.example config/aura.env` creates the file that step 4 loads), create the `Anygrasp` env (`conda env create -f environments/anygrasp.yml`), and run the code block of [step 4](#4-anygrasp-sdk-license-and-checkpoint), which clones the SDK and prints the feature id to register.

### 1. Clone and create your configuration

```bash
git clone https://github.com/Yoonkyo/AuRA.git && cd AuRA
cp config/aura.env.example config/aura.env   # filled in at step 6; the install steps already read its default paths
```

To install Grounded-SAM-2, its weights or the AnyGrasp SDK outside the repository, set `AURA_GSAM_REPO`, `AURA_GSAM_CHECKPOINTS` or `AURA_ANYGRASP_SDK` in `config/aura.env` now: steps 3 and 4 install into these paths. Write absolute paths, as `$HOME/...` rather than `~/...`, because bash does not expand `~` inside the file's quotes.

### 2. Conda environments

```bash
conda env create -f environments/aura.yml       # env "AuRA"
conda env create -f environments/anygrasp.yml   # env "Anygrasp"; skip if you created it for the license request
```

conda may ask you once to accept the Anaconda Terms of Service for the `defaults` channel, which both files use. Accept at the prompt, or run the `conda tos accept` commands it prints and start `conda env create` again.

The header of each file lists numbered **POST-INSTALL STEPS** that a conda yml cannot express ("header steps" below, to tell them from the steps of this page). Run header steps 1-2 of each file now, inside that file's activated env, after `set -a; source config/aura.env; set +a`; the later header steps belong to steps 3 and 4 of this page:

- [`environments/aura.yml`](../environments/aura.yml): header steps 1-2 install `supervision` and PyRoki (pinned commit) with `--no-deps`. Header step 3, Grounded-SAM-2, is [step 3](#3-grounded-sam-2-and-model-weights) of this page. The header ends with a sanity check to run after that.
- [`environments/anygrasp.yml`](../environments/anygrasp.yml): header step 1 installs `graspnetAPI` with `--no-deps`, and header step 2 builds AnyGrasp's MinkowskiEngine fork against CUDA 12.8. Header steps 3-6 (SDK and `pointnet2`, ABI-specific binaries, license, checkpoint) are [step 4](#4-anygrasp-sdk-license-and-checkpoint) of this page.

The `AuRA` env must contain exactly one OpenCV, `opencv-contrib-python==4.7.0.72`, because the calibration tools need both ChArUco APIs. Never install anything into it that pulls in another OpenCV wheel (`opencv-python`, `opencv-python-headless`, ...): they all install into the same `cv2/` directory. For the check and the fix, see [the opencv-contrib hazard](../environments/README.md#the-opencv-contrib-470-hazard). Versions, rationale and known `pip check` messages are in [environments/README.md](../environments/README.md).

### 3. Grounded-SAM-2 and model weights

```bash
conda activate AuRA
set -a; source config/aura.env; set +a
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git "$AURA_GSAM_REPO"   # default third_party/Grounded-SAM-2
git -C "$AURA_GSAM_REPO" checkout b7a9c29f196edff0eb54dbe14588d7ae5e3dde28
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"   # a CUDA 12.8 toolkit with nvcc
export PATH="$CUDA_HOME/bin:$PATH"
SAM2_BUILD_CUDA=0 pip install --no-build-isolation --no-deps -e "$AURA_GSAM_REPO"
pip install --no-build-isolation --no-deps -e "$AURA_GSAM_REPO/grounding_dino"

mkdir -p "$AURA_GSAM_CHECKPOINTS"                  # default checkpoints/groundedSAM
wget -P "$AURA_GSAM_CHECKPOINTS" https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
wget -P "$AURA_GSAM_CHECKPOINTS" https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cp "$AURA_GSAM_REPO/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py" "$AURA_GSAM_CHECKPOINTS/"
```

Check the downloads against the sha256 sums in [environments/README.md](../environments/README.md#model-weights). Two more assets download automatically the first time they are used, so the machine needs network access once:

- `bert-base-uncased`, GroundingDINO's text encoder;
- the `franka_description` robot model, used by the PyRoki filter.

To fetch both in advance, see [First-run downloads](../environments/README.md#first-run-downloads).

### 4. AnyGrasp SDK, license and checkpoint

The AnyGrasp SDK, its license files and its checkpoint are **not part of this repository** and cannot be redistributed. You obtain them from the AnyGrasp authors under their terms.

The license is **machine-locked**: its feature id is computed from the MAC address(es) of the active wired network interfaces. Connect the rig's final network hardware (for example the adapter for the robot network) before you generate the id. `license_checker` needs `ifconfig` and `iwconfig` (`sudo apt install net-tools wireless-tools`) and the env's OpenSSL 1.1, hence the `LD_LIBRARY_PATH` below. To clone the SDK and print this machine's feature id (skip the `git clone` and `git checkout` lines if you ran them for the license request):

```bash
conda activate Anygrasp
set -a; source config/aura.env; set +a
git clone https://github.com/graspnet/anygrasp_sdk.git "$AURA_ANYGRASP_SDK"   # default third_party/anygrasp_sdk
git -C "$AURA_ANYGRASP_SDK" checkout 3cb4a593e1a7cdde6ac897cfce49c26e1c41db9c
(cd "$AURA_ANYGRASP_SDK/license_registration" && LD_LIBRARY_PATH="$CONDA_PREFIX/lib" ./license_checker -f)   # prints the feature id
```

Keep the SDK folder out of any repository you publish: it will contain your license. Then follow [anygrasp_license.md](anygrasp_license.md#3-request-the-license) from section 3 on: register the feature id; while you wait for the reply, build `pointnet2` and stage the `gsnet.so` / `lib_cxx.so` binaries (header steps 3-4 of `environments/anygrasp.yml`, without the `git` lines of header step 3: the SDK is already cloned); then install and verify the license, place `checkpoint_detection.tar` and run the sanity check. `scripts/anygrasp_poses_one_shot.sh` sets up the AnyGrasp runtime itself (`LD_LIBRARY_PATH`, `PYTHONPATH`, and the working directory, from which the license is resolved).

### 5. Robot environment

```bash
uv sync --project environments/robot --python 3.12   # Python 3.12 as on the authors' machine; 3.11 also works
uv run --project environments/robot python -c "import fr3, pylibfranka, depthai, cv2, pygame"   # import check, no hardware access
uv run --project environments/robot python aura/robot/gamepad_record_franka_3oak.py --help   # prints the recorder's options
```

The robot steps are the scripts in `aura/robot/`, run as `uv run --project environments/robot python aura/robot/<script>.py`, the final release-and-home step of `scripts/run_anygrasp_to_lift_one_shot.sh` and the end-effector pose read of the fingertip re-fit ([calibration.md](calibration.md#re-fit), step 3). They need the **PREEMPT_RT kernel**: `pylibfranka` checks for it, and for real-time scheduling rights, when it connects to the arm (`RealtimeConfig.kEnforce`). Check that `cat /sys/kernel/realtime` prints `1`, `ulimit -r` prints `99` and `ulimit -l` prints `unlimited`; if not, set up the kernel, the NVIDIA driver for it and your user's real-time rights as in [hardware.md](hardware.md#real-time-kernel). Run robot scripts by hand from a shell without an active conda env, so that no conda `LD_LIBRARY_PATH` leaks into the real-time process ([hardware.md, Shell](hardware.md#real-time-kernel)). See also [Robot environment](../environments/README.md#robot-environment-robot) and [third_party/fr3/README.md](../third_party/fr3/README.md).

### 6. Configure `config/aura.env`

Set your rig's values in `config/aura.env`. First install the Movidius udev rule for the cameras, if it is not installed yet:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Then list the MXIDs of the connected cameras in the `AuRA` env:

```bash
python -c "import depthai as dai; [print(d.getDeviceId()) for d in dai.Device.getAllAvailableDevices()]"
```

Finally, fill in the values:

```bash
: "${AURA_ROBOT_IP:=<ROBOT_IP>}"                  # FCI address of your FR3 (required by every robot stage)
: "${AURA_OAK_IDS:=<MXID_0> <MXID_1> <MXID_2>}"   # camera index 0, 1, 2 (required)
: "${AURA_HANDEYE_OAK_ID:=oak_<MXID_0>}"          # optional: camera used for hand-eye calibration
```

The order of `AURA_OAK_IDS` defines camera indices 0, 1 and 2 in every recording (`color/<i>`, `depth/<i>`, the order of `calibrate.pkl`), so keep it fixed once you have data. Calibration files are named by MXID (`oak_<MXID>`). To match the MXIDs to the physical cameras and to choose the hand-eye camera, see [hardware.md](hardware.md#cameras).

The remaining entries have working defaults: the conda env names (`AURA_ENV=AuRA`, `ANYGRASP_ENV=Anygrasp`) and the paths `AURA_ROBOT_PROJECT`, `AURA_DATA_ROOT`, `AURA_CALIB_ROOT`, `AURA_GSAM_CHECKPOINTS`, `AURA_GSAM_REPO` and `AURA_ANYGRASP_SDK`, which point inside the repository. If you change one, write an absolute path, using `$HOME/...` rather than `~/...` ([step 1](#1-clone-and-create-your-configuration)). If you change `AURA_GSAM_CHECKPOINTS`, `AURA_GSAM_REPO` or `AURA_ANYGRASP_SDK` after steps 3 and 4, move the weights, the clone or the SDK (with its license and checkpoint) to the new path. A moved Grounded-SAM-2 clone also needs the two `pip install` commands of [step 3](#3-grounded-sam-2-and-model-weights) again, after that step's `conda activate`, `source` and `export` lines, because its editable installs record the old path. Only `scripts/run_anygrasp_to_lift_one_shot.sh` reads `AURA_ROBOT_PROJECT`, for its robot stages (`--uv-project` overrides it). The robot commands in the documentation use its default, `environments/robot`; if you point it elsewhere, use the same path in those commands.

- **After you edit `config/aura.env`, open a new terminal** (or run `unset ${!AURA_*} ANYGRASP_ENV`) before you load it again or start a script. Loading the file exports its values, and a non-empty value already exported in your shell takes precedence over the file, except `AURA_REPO_ROOT`, which the scripts in `scripts/` set to their own checkout. A shell that has loaded it (steps 2 to 4 did) therefore keeps the old values, also in every script started from it.
- To run a Python tool directly, load the file from bash with `set -a; source config/aura.env; set +a`. From another shell, export `AURA_REPO_ROOT=<path to the repository>` first, or the file stops with an error.

## Next step: calibrate your rig

Every rig needs its own calibration: camera intrinsics, then extrinsics, then hand-eye, plus the fingertip offsets. The procedure, the quality targets and the tool options are in [calibration.md](calibration.md). After that, continue with [usage.md](usage.md).
