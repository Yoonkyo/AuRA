# Hardware and system requirements

This page lists what your rig and workstation need for AuRA v1, the automated data collection described in the [README](../README.md). Installation is described in [installation.md](installation.md) and calibration in [calibration.md](calibration.md). Post-processing ([`aura/postprocessing/`](../aura/postprocessing/README.md)) and training and asset reconstruction ([`aura/reconstruction/`](../aura/reconstruction/README.md)) are coming soon (v2); their requirements will be documented with them.

| Component | Requirement | Needed for |
|---|---|---|
| Robot | Franka Research 3 with the Franka Hand (standard fingers) and the Franka Control Interface (FCI) | grasp execution, teleoperation |
| Cameras | 3 × Luxonis OAK-D, each on its own USB 3 port, rigidly mounted around the workspace | every recording |
| Calibration target | printed ChArUco board, 4 × 5 squares of 40 mm, 30 mm markers, `DICT_4X4_50` | camera intrinsics, extrinsics, hand-eye calibration and re-anchoring |
| Input device | PlayStation-style gamepad | teleoperation for hand-eye and fingertip calibration |
| Workstation | one Linux x86-64 PC with an NVIDIA GPU and a PREEMPT_RT kernel | everything |

## Workstation

One workstation runs everything: perception, GPU inference and the 1 kHz robot control. The pipeline stages run one after another.

| Item | Requirement | Notes |
|---|---|---|
| OS | Linux x86-64, glibc ≥ 2.34 | The authors use Ubuntu 24.04. `pylibfranka` 0.20.5 needs glibc ≥ 2.34 and `depthai` 3.6.1 needs ≥ 2.28. |
| Kernel | PREEMPT_RT, for the robot steps only | See [Real-time kernel](#real-time-kernel). |
| RAM | 32 GiB or more | Recordings stay in RAM until they end (see below). The authors' machine has 62 GiB. |
| GPU | NVIDIA, with a driver that supports CUDA 12.8 (≥ 570) | Used by Grounded-SAM-2 (stage 02) and AnyGrasp (stage 03). The authors use an RTX 5090 (32 GB, sm_120). The pinned `torch 2.11.0+cu128` wheels cover sm_75 to sm_120. The minimum GPU memory was not measured. |
| CUDA toolkit | 12.8 with `nvcc`, found through `CUDA_HOME` | Needed only to build `groundingdino._C`, MinkowskiEngine and `pointnet2`. |
| Disk | a few tens of GB | About 2.3 GB of model weights, about 9 GB per conda env, about 0.6 GB per robot-executed case (0.57 to 0.84 GB in the authors' 26 cases with the default settings). A raw synchronized frame takes about 6 MB for all three cameras at 1280 × 720. |
| Display | a desktop session | Stage 03 renders its debug images through a hidden Open3D window, and the `--preview` options open camera windows. |
| Network | a wired link to the robot, plus internet access during installation and first use | pip, git clones, model weights, and the first-run downloads `bert-base-uncased` and `franka_description` ([First-run downloads](../environments/README.md#first-run-downloads)). |
| Packages | conda, [uv](https://docs.astral.sh/uv/), `build-essential`, `git`, `net-tools`, `wireless-tools`, and `ninja-build` (recommended) | The AnyGrasp license tool calls `ifconfig` and `iwconfig`. Without `ninja-build`, the MinkowskiEngine build takes about 7 minutes instead of about 1. |

**RAM.** The recorders estimate the memory a recording needs as width × height × 5 bytes × 3 cameras × fps × seconds × 1.6, print the estimate and warn when it exceeds 75 % of `MemAvailable`. At 1280 × 720 and 15 fps this is about 0.31 GiB per second: 9.3 GiB for 30 s and 18.5 GiB for 60 s. At 640 × 400 it is about 2.6 GiB for 30 s. The recording executor estimates for its `--record-max-duration` (60 s), although a lift recording lasts only a few seconds.

## Real-time kernel

The robot steps run in the uv project `environments/robot`:

```bash
uv run --project environments/robot python aura/robot/<script>.py ...
```

These steps are the recording executor `aura/robot/execute_anygrasp_franka_record_lift.py` (stage 02 of the lift one-shot, `scripts/run_anygrasp_to_lift_one_shot.sh`), the standalone executor `aura/robot/execute_anygrasp_franka.py`, the teleoperation recorder `aura/robot/gamepad_record_franka_3oak.py`, the lift one-shot's stage `02b_release_and_return_home`, and the end-effector pose read of the fingertip re-fit ([calibration.md](calibration.md#re-fit), step 3). `pylibfranka` creates its robot connection with `RealtimeConfig.kEnforce`, so connecting to the arm fails with a real-time error unless the kernel is PREEMPT_RT and the process may use the highest real-time priority, also for a step that only reads the pose or opens the gripper. Every other step (capture, calibration fitting, segmentation, AnyGrasp, the PyRoki filter) is an ordinary process and runs on any kernel.

Check the machine:

```bash
uname -v                    # contains PREEMPT_RT
cat /sys/kernel/realtime    # 1
ulimit -r                   # 99
ulimit -l                   # unlimited
```

- **Kernel.** On Ubuntu, the `linux-realtime` kernel is available through Ubuntu Pro; you do not need to patch and build one yourself.
- **GPU driver.** NVIDIA does not support PREEMPT_RT kernels, and its kernel module refuses to build for one (the DKMS `make.log` says `The kernel you are installing for is a PREEMPT_RT kernel!`). A session still runs the GPU stages under the real-time kernel (the lift one-shot runs segmentation and AnyGrasp for a new case in the same call as the robot steps), so build the module for the real-time kernel with that check disabled, for example through DKMS (the driver's `nvidia-dkms-*` package on Ubuntu): `sudo IGNORE_PREEMPT_RT_PRESENCE=1 dkms autoinstall -k <real-time kernel version>`, with the version as `uname -r` prints it under that kernel. The authors' workstation runs `nvidia-dkms-580-open` 580.126.09, built by DKMS for `linux-realtime` with this check disabled. Repeat the build after every driver or kernel update, then boot the real-time kernel and check that `nvidia-smi` lists the GPU.
- **Without PREEMPT_RT.** As shipped, the robot scripts cannot connect to the arm on a kernel without PREEMPT_RT. libfranka's alternative, `pylibfranka.RealtimeConfig.kIgnore` as the second argument of `Robot(...)`, is a code change in three places: the `Robot(ip)` call in `third_party/fr3/src/fr3/robot.py` and the `Robot(...)` calls in `aura/robot/execute_anygrasp_franka_record_lift.py` and `aura/robot/gamepad_record_franka_3oak.py`. libfranka then no longer checks `/sys/kernel/realtime`; it still requests the highest `SCHED_FIFO` priority, but carries on at normal priority if the process lacks the rights of the next bullet. AuRA has never been run that way.
- **Privileges.** Give your user real-time rights, for example through a `realtime` group. Create the group and add yourself once with `sudo groupadd -f realtime && sudo usermod -aG realtime "$USER"`, then put these two lines into a file in `/etc/security/limits.d/`, for example `99-realtime.conf`:

  ```
  @realtime  -  rtprio   99
  @realtime  -  memlock  unlimited
  ```

  Log out and in again (reboot if `ulimit -r` still prints `0`). A container needs the same rights.
- **Shell.** No conda `LD_LIBRARY_PATH` may leak into the real-time process:
  - Run robot scripts by hand from a shell without an active conda env: repeat `conda deactivate` until `echo "${CONDA_SHLVL:-0}"` prints `0` (twice after `conda activate AuRA` in a shell that started in `(base)`).
  - **Exception:** start the lift one-shot with the `AuRA` env active. It uses that env's `python` for its own bookkeeping and calls `uv run` for the robot steps.
  - `conda deactivate` does not undo an `LD_LIBRARY_PATH` exported in the shell, such as the one of the AnyGrasp check in [anygrasp_license.md](anygrasp_license.md#7-check-and-run). After such an export, start the robot scripts and the lift one-shot from a new terminal.

See also [Robot environment](../environments/README.md#robot-environment-robot) and [`third_party/fr3/README.md`](../third_party/fr3/README.md).

## Robot

- **Arm and hand.** A Franka Research 3 with the Franka Hand. The robot system version must be compatible with libfranka 0.20.5, which is bundled in `pylibfranka` 0.20.5.
- **End-effector frame.** Keep the end-effector configuration in Franka Desk at the Franka Hand default. The executors command libfranka's configured end-effector frame (`O_T_EE`), the grasp filter plans for `fr3_hand_tcp` of `franka_description`, and the fingertip offsets are expressed in that frame. All three must coincide. With the Franka Hand default, the end-effector frame lies 0.1034 m along the flange's z axis and is rotated by -45° about that axis (`F_T_EE` in libfranka's robot state). Nothing in the pipeline checks this: with a different frame, the executors move the fingertips to poses that the filter never checked.
- **Fingers.** The grasp filter models the gripper by two calibrated points, the outer tips of the fingers. The shipped [`assets/franka_fingertip_table_calibration_simple.json`](../assets/README.md) describes the authors' Franka Hand, 104.2 mm from outer tip to outer tip with the gripper open. Re-fit it if your fingers or pads differ ([fingertip geometry](calibration.md#4-fingertip-geometry)).
- **Network and FCI.** Connect the robot's FCI network port to the workstation with a wired link. Put its address in `AURA_ROBOT_IP` in `config/aura.env`. The robot scripts also accept it on the command line: `--robot-ip` for the lift one-shot, `--ip` for the executors, and the first positional argument for the teleoperation recorder. Give the workstation's interface on the wired link a static IPv4 address in the robot's subnet. Before each session, check that `ping -c 3 <ROBOT_IP>` answers ([pre-flight checklist](usage.md#pre-flight-checklist)): the lift one-shot checks only that a robot address is set, so on a new case an unreachable robot stops the run at the robot step, after the capture, segmentation, AnyGrasp and the filter. If you add a USB-Ethernet adapter for the robot network, do so before you request the AnyGrasp license ([why](anygrasp_license.md#3-request-the-license)).
- **Every session.** In Franka Desk (`https://<ROBOT_IP>/desk/`), unlock the joints and activate FCI. Nothing can connect otherwise.
- **One controller at a time.** Only one program can control the arm. The gripper uses a separate connection (port 1338). Quit the teleoperation recorder before you start an executor, and the other way round.
- **Home pose.** Every robot script that homes the arm moves it to `fr3.HOME_Q` = `[0, -π/4, 0, -3π/4, 0, π/2, π/4]`. The grasp filter plans the motion from the joint state in its `--state-file`; the preparation script and the lift one-shot pass [`assets/fr3_home_state.npz`](../assets/README.md), measured at this pose, unless you give another. **If the arm does not start the grasp motion from that state, it moves along a path the filter never checked:**
  - Lift one-shot: it passes `--home` to the recording executor by default. Its `--no-home` requires an explicit, freshly captured `--state-file`. On a prepared case, also keep the default `--refilter` (the script does not check this): with `--no-refilter` (or `REFILTER=0`) the preparation skips the filter, and the arm moves from its current pose along a path checked from the old state file.
  - Executors run by hand: they home only when you pass `--home`, which is off by default. With `--execute`, pass `--home` unless the filter's `--state-file` was captured at the arm's current pose.
  - Another home configuration: if your robot settles at a different configuration, capture a new state file yourself (no script in this repository writes one): an `.npz` whose key `q` holds the 7 joint angles in rad after homing (format in [`assets/README.md`](../assets/README.md)). Pass it as an absolute `--state-file`.
- **Motion.** A grasp is executed as straight Cartesian moves in the robot base frame: from home to 10 cm above the grasp, down to the grasp, and 15 cm up. The filter checks this path for IK, joint limits, table clearance of the two fingertips and distance to the cameras. It does not model obstacles, the arm links or the hand body, so keep the area around the object clear.
- **Safety.** Keep the emergency stop within reach. The AuRA robot scripts set the same collision thresholds and impedances as the `fr3` wrapper: torque thresholds 20 / 40 Nm (lower / upper), force thresholds 10 / 20 N. The executors close the gripper towards zero width with 35 N (`--grasp-force`), the teleoperation recorder with a fixed 20 N, so deformable objects are squeezed firmly.

## Cameras

- **Model and connection.** Three Luxonis OAK-D cameras, used through the DepthAI v3 API (`depthai` 3.6.1). Connect each one to its own USB 3 port; all recorders print `USB speed=` for each camera at start-up. A camera that falls back to USB 2 delivers fewer frames, and the offline synchronization then drops timesteps. Linux needs the Movidius udev rule ([System requirements](../environments/README.md#system-requirements)).
- **IDs and order.** List the MXIDs of the connected cameras in the `AuRA` env:

  ```bash
  python -c "import depthai as dai; [print(d.getDeviceId()) for d in dai.Device.getAllAvailableDevices()]"
  ```

  Put them in `AURA_OAK_IDS` in `config/aura.env`, separated by spaces. Their order defines camera indices 0, 1 and 2 in every recording (`color/<i>/`, `depth/<i>/`, `calibrate.pkl`). Keep the order fixed once you start recording. Calibration files are named `oak_<MXID>`.
- **Which camera is which.** In the `AuRA` env, load the file with `set -a; source config/aura.env; set +a` and run `python aura/capture/manual_align_3oak.py` (or pass the MXIDs with `--oak-ids`). It shows the three live views side by side, each labelled `oak_<MXID>`, and records nothing. Use it to aim the cameras and to match MXIDs to physical cameras. Press q or Esc to quit. If you change the order in `config/aura.env` afterwards, open a new terminal (or `unset AURA_OAK_IDS`) before you load it again.
- **Hand-eye camera.** Choose the camera with the best view of the space in front of the robot. During hand-eye calibration it must see the board held in the gripper over a wide range of poses. Set `AURA_HANDEYE_OAK_ID=oak_<MXID>`, with the `oak_` prefix (unlike the bare MXIDs in `AURA_OAK_IDS`).
- **Placement.**
  - All three cameras must see the board lying flat on the table where the world origin will be, and the objects you want to record.
  - The fusion step uses depth between 0.2 m and 1.5 m from a camera only.
  - The grasp filter rejects grasps that bring the gripper within 0.12 m of a camera centre (`--camera-keepout-radius-m`), so keep the cameras out of the arm's way. The authors' cameras are 0.40 to 0.46 m above the table.
  - Mount the cameras rigidly. Moving a camera invalidates its extrinsic calibration ([re-anchoring](calibration.md#5-optional-re-anchor-moved-cameras)).
- **Streams.** There are two presets: `low` = 640 × 400 and `high` = 1280 × 720. Every pipeline stage records at `high` by default, except the teleoperation recorder (default `low`, so pass `--resolution high`). RGB and depth are requested at the same size and depth is aligned to RGB on the device, so `rgb.jpg` and `depth.npy` are pixel-aligned and share one intrinsic matrix. Intrinsics are calibrated per camera and resolution. All recorders default to 15 fps.
- **Undistortion.** The cameras undistort RGB on the device (`--enable-undistortion True`, the default of every recorder). The intrinsics are fitted on these undistorted frames, so the stored distortion coefficients are only a small residual. Never undistort the images a second time, and never change the flag after calibrating.
- **Synchronization.** There is no hardware trigger. The recorders buffer every frame of every camera in RAM with its capture timestamp. After recording, they search 120 phases of a fixed 15 Hz grid for the one that keeps the most timesteps and match each camera's nearest frame to each grid time. A timestep is dropped when any camera has no frame within the tolerance: 45 ms for dynamic recordings, 70 ms for the static snapshot of the preparation run, and 75 ms as used for the authors' calibration captures. The recording executor and the teleoperation recorder match on these capture timestamps. `aura/capture/record_3oak.py` (static snapshot, calibration captures, human baseline) matches on the capture timestamps plus a fixed receive latency per camera, measured during the warm-up (`projectors.<camera>.fixed_offset` in `meta.json`). The latencies differ between cameras, so one human-baseline bundle can hold frames captured one to three frame periods apart ([Known limitations](troubleshooting.md#recording-and-cameras)); static captures are not affected.

## ChArUco board

| Property | Value |
|---|---|
| Squares | 4 (X) × 5 (Y) |
| Square side | 40 mm (`--square-length 0.04`) |
| Marker side | 30 mm (`--marker-length 0.03`) |
| Dictionary | `DICT_4X4_50` |
| Board size | 160 × 200 mm, 10 markers, 12 inner corners |

The calibration tools take the board geometry as flags (`--squares-x 4 --squares-y 5 --square-length 0.04 --marker-length 0.03 --dictionary DICT_4X4_50`); `aura/calibration/calibrate_3oak_anchor108.py` has it built in. To generate a printable image (1600 × 2000 px, 10 px per mm) in the `AuRA` env:

```bash
python -c "import cv2; b = cv2.aruco.CharucoBoard((4, 5), 0.04, 0.03, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)); cv2.imwrite('charuco_4x5_40mm.png', b.generateImage((1600, 2000)))"
```

- **Print** at 100 % scale (254 dpi for this image, no "fit to page") and measure a square: it must be 40.0 mm. Reprint rather than adapting the flags, because of the built-in dimensions.
- **Layout.** The board has an odd number of rows, so OpenCV's legacy (before 4.6) and current board layouts are identical for it, and a board printed with any OpenCV version works.
- **Mounting.** Glue the print flat on a rigid, thin backing. For the extrinsic calibration it lies flat on the table. For the hand-eye calibration the gripper must hold it rigidly, for example by a tab that the fingers clamp. Any fixed offset between the gripper and the board is estimated.
- **Frame.** The origin is the outer corner of the first square, which is the top-left corner of the generated image. +X runs along the 4-square (160 mm) edge, +Y along the 5-square (200 mm) edge, and +Z = X × Y points into the board. With the board face-up on the table, +Z points down into the table, and this frame becomes AuRA's world frame ([calibration.md](calibration.md#the-world-frame)). The plane z = 0 is the board's top surface: a thick backing places the real table slightly below z = 0, which only makes the table-clearance checks more conservative.

## Gamepad

A PlayStation-style (DualShock) controller that `pygame` can read; the first joystick found is used. The mapping uses the Linux `pygame` indices of such a pad: Cross 0, Circle 1, Square 3, L1 4, R1 5, the triggers on axes 2 and 5, and the D-pad as hat 0. Other pads may map differently. The controls are listed in [calibration.md](calibration.md#teleoperation-controls).

Before the first teleoperation session, check the triggers as the recorder reads them, with `pygame` on axes 2 and 5 (other joystick tools use other scales). From the repository root, with both triggers released, run:

```bash
uv run --project environments/robot python -c "import time, pygame; pygame.init(); j = pygame.joystick.Joystick(0); j.init(); time.sleep(1); pygame.event.pump(); print(j.get_name(), 'L2:', round(j.get_axis(2), 2), 'R2:', round(j.get_axis(5), 2))"
```

Both values must be -1.0. The recorder maps a trigger value of -1.0 to "no motion"; a trigger that rests at 0.0 commands z or yaw motion as soon as teleoperation starts.

## Before every session

Run the [pre-flight checklist](usage.md#pre-flight-checklist) in usage.md before every session.

Problems and their fixes are collected in [troubleshooting.md](troubleshooting.md).
