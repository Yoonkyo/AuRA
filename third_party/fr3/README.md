# fr3 — High-Level Control for Franka Research 3

High-level Python library for the Franka Research 3 (FR3) robot. It wraps `pylibfranka` to provide clean Cartesian pose control, joint target tracking, velocity streaming, and gripper operations.

## Quick Start

The library uses a module-level configuration for the robot IP address, similar to an API key pattern. The examples below read it from the `AURA_ROBOT_IP` environment variable, so load `config/aura.env` into your bash shell first: from the repository root, run `set -a; source config/aura.env; set +a` (the scripts in `scripts/` do this themselves). If `AURA_ROBOT_IP` is unset or empty, `FR3()` raises `No robot IP configured`.

```python
import fr3
import os
fr3.ROBOT_IP = os.environ.get("AURA_ROBOT_IP") or None  # FCI address of your robot (config/aura.env)
# or pass it explicitly: fr3.FR3(ip="<ROBOT_IP>")

robot = fr3.FR3()
robot.home(blocking=True)
pose = robot.get_pose()
print(f"Position: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})")
```

### Requirements

- Linux x86_64 with CPython 3.11 or 3.12. `fr3` requires Python 3.11 or newer, and `pylibfranka` 0.20.5 has no wheel for CPython 3.13 or newer (its x86_64 Linux wheels for CPython 3.10-3.12 need glibc 2.34 or newer, e.g. Ubuntu 22.04+). `fr3` also imports the POSIX-only `fcntl` module.
- A PREEMPT_RT kernel and an account with real-time scheduling rights (the Franka installation guide grants these through a `realtime` group with `rtprio 99` and raised `memlock` limits), also for an `FR3()` that only reads the state or moves the gripper: `FR3()` creates `pylibfranka.Robot(ip)` with pylibfranka's default `RealtimeConfig.kEnforce`, so it raises `pylibfranka.RealtimeException` (a `RuntimeError`) when it connects unless `/sys/kernel/realtime` is `1` and the process can switch to the highest `SCHED_FIFO` priority.
- The Franka Control Interface (FCI) enabled in Franka Desk, a network route to the robot's FCI address, and a robot system version compatible with libfranka 0.20.5 (bundled in `pylibfranka` 0.20.5).

### Installation

In AuRA, `fr3` is installed as a path dependency of `environments/robot` (editable, see `[tool.uv.sources]` in `environments/robot/pyproject.toml`). From the repository root:

```bash
uv sync --project environments/robot
```

Scripts that import `fr3` run in that environment: `uv run --project environments/robot python <script>`.

## Coordinate Frames and Conventions

- **Base Frame:** All end-effector poses are defined in the robot base frame.
- **Orientation:** Uses ZYX intrinsic Euler angles. Users provide `(roll, pitch, yaw)` in radians. `(0, 0, 0)` is the base frame's orientation (end-effector z axis pointing up); with the end-effector configuration in Franka Desk at the Franka Hand default, `get_pose()` returns roll ≈ ±π, pitch ≈ 0, yaw ≈ 0 at `HOME_Q` (hand pointing down).
- **Home Position:** The default home configuration is `fr3.HOME_Q = [0, -π/4, 0, -3π/4, 0, π/2, π/4]` (radians, joints 1-7). AuRA's `assets/fr3_home_state.npz`, the starting arm configuration for grasp filtering, was measured at this pose.

## API Reference

### Constructor and Context Manager

#### `FR3(ip=None)`
Connects to the robot. If `ip` is not provided, it uses `fr3.ROBOT_IP`. Supports context manager usage for automatic cleanup.

```python
with fr3.FR3() as robot:
    robot.home(blocking=True)
```

### State Information

#### `get_pose() -> Pose`
Returns the current end-effector pose as a `Pose` object containing `x, y, z, roll, pitch, yaw`.

#### `get_joints() -> np.ndarray`
Returns a 7-element numpy array of current joint angles in radians.

### Motion Control

#### `move_to_pose(x, y, z, roll, pitch, yaw, speed=0.3, blocking=False)`
Moves the end-effector to the target pose using Cartesian interpolation.
- `speed`: Fraction 0.0 to 1.0.

#### `home(speed=0.3, blocking=False)`
Moves the robot to its predefined home joint configuration.

#### `set_velocity(vx, vy, vz, wx, wy, wz, smoother_params=None)`
Streams Cartesian velocity commands for linear (m/s) and angular (rad/s) motion. Useful for joystick or teleoperation control.
- `smoother_params`: Optional dict for the 1 kHz jerk-limited smoother, applied when the velocity thread starts (first call or after recovery). Keys and defaults: `K=15.0`, `max_accel=[6.0]*3 + [12.0]*3` (m/s², rad/s²), `max_jerk=[0.04]*3 + [0.08]*3` (acceleration change per 1 ms tick, tripled while decelerating), `command_timeout=0.5` (seconds without a new command before the target velocity is treated as zero).

#### `set_joint_targets(targets, speed=0.3, blocking=False)`
Moves joints to target angles using S-curve trajectories. Supports multiple speed formats (see Speed Formats section).

#### `start_streaming(speed=0.3, drop_last=0) -> StreamingMotionGenerator`
Starts streaming joint position control. Returns a `StreamingMotionGenerator` that accepts waypoints via `add_waypoints()`. The robot holds its current position until waypoints are provided. Call `stop()` to end streaming mode.

- `drop_last`: When >0, each `add_waypoints()` call removes up to this many waypoints from the tail of the queue before appending new ones. Enables MPC-style rolling-horizon replanning where speculative waypoints are replaced by fresher predictions.

The generator also exposes:
- `last_completed_waypoint`: Joint angles of the most recently passed-through waypoint (or `None`).
- `last_completed_index`: Zero-based index of that waypoint (-1 if none completed).

#### `send_joint_waypoints(waypoints)`
Sends joint waypoints to an active streaming controller. `waypoints` is shape `(7,)` or `(N, 7)`. Raises `RuntimeError` if streaming mode is not active.

### Motion Management

- `is_moving() -> bool`: Returns True if a motion is currently active.
- `wait(timeout=None) -> bool`: Blocks until the current motion completes or timeout expires.
- `stop()`: Immediately stops all active motion.
- `recover()`: Recovers the robot from reflex errors or collision states.

### Gripper Control

Gripper operations use a separate TCP connection on port 1338 and can be called during arm motion.

- `move_gripper(width, speed=0.1)`: Moves the gripper to a target width in meters.
- `open_gripper(speed=0.1)`: Opens the gripper fully.
- `close_gripper(speed=0.1)`: Closes the gripper fully.
- `grasp(width, speed=0.1, force=20.0, epsilon_inner=0.005, epsilon_outer=0.005)`: Grasps an object at a target width with specified force and tolerances.
- `homing_gripper()`: Runs the gripper homing sequence.

## Blocking vs Non-Blocking

Most motion methods include a `blocking` parameter:

- **`blocking=False` (default):** The motion starts in a background thread, and the function returns immediately. This allows for asynchronous execution or state monitoring during movement.
- **`blocking=True`:** The function blocks execution until the motion is completed.

Example of manual blocking using `wait()`:

```python
robot.move_to_pose(0.5, 0.0, 0.4, 3.14159, 0, 0) # Starts moving (roll ≈ π: hand pointing down, as at home)
# ... perform other tasks ...
robot.wait() # Now block until arrival
```

## Speed Formats

The `set_joint_targets` method supports three distinct ways to specify motion speed. Each sets a speed factor that scales the S-curve's velocity limits (2.0 rad/s for joints 1-4, 2.5 rad/s for joints 5-7) and its acceleration limit (5.0 rad/s² for every joint): `speed` is the factor itself, while a scalar `speed_rad_s` and `duration` are converted into one, clipped to 0.01-1.

| Format | Parameter | Example |
|--------|-----------|---------|
| Fraction (0-1) | `speed=0.3` | `robot.set_joint_targets(q, speed=0.3)` |
| Rad/s | `speed_rad_s=1.0` | `robot.set_joint_targets(q, speed_rad_s=1.0)` |
| Requested duration (seconds) | `duration=3.0` | `robot.set_joint_targets(q, duration=3.0)` |

Note: Provide exactly one speed parameter per call. A scalar `speed_rad_s` is the velocity limit of joints 5-7; joints 1-4 get 0.8 times it. `duration` is not the motion time: requests longer than about 1.3 s finish early (`duration=3.0` takes about 2.0-2.2 s, `duration=10.0` about 4.9-5.8 s, depending on which joint moves furthest), and shorter ones take longer. This assumes the largest joint change is at least about `duration`/100 rad; smaller moves hit the speed factor's 0.01 floor and finish sooner (a 0.05 rad move with `duration=10.0` takes about 2.8-3.1 s).

## Policy Deployment Examples

### Simple (blocking per step)

```python
import fr3
import numpy as np
import os

fr3.ROBOT_IP = os.environ.get("AURA_ROBOT_IP") or None

with fr3.FR3() as robot:
    robot.home(blocking=True)
    
    for step in range(100):
        obs = robot.get_joints()
        action = policy.predict(obs)  # Predict next joint angles
        robot.set_joint_targets(action, speed=0.5, blocking=True)
```

### Streaming (continuous motion from batched waypoints)

For policies that output batches of future joint configurations, streaming mode provides smooth, uninterrupted motion. The robot interpolates through all waypoints using quintic polynomials, decelerates at the end, and seamlessly resumes when new waypoints arrive.

```python
import fr3
import numpy as np
import os
import time

fr3.ROBOT_IP = os.environ.get("AURA_ROBOT_IP") or None

with fr3.FR3() as robot:
    robot.home(blocking=True)

    stream = robot.start_streaming(speed=0.4)

    for step in range(100):
        obs = robot.get_joints()
        waypoints = policy.predict_batch(obs)  # (N, 7) array

        while stream.waypoints_remaining > 2:
            time.sleep(0.01)

        stream.add_waypoints(waypoints)

    robot.stop()
```

### MPC-style streaming (rolling horizon with speculative waypoints)

For policies that plan long horizons but only commit a portion, `drop_last` replaces speculative tail waypoints when a new batch arrives.

```python
import fr3
import numpy as np
import os
import time

fr3.ROBOT_IP = os.environ.get("AURA_ROBOT_IP") or None
HORIZON, DROP_LAST = 16, 8

with fr3.FR3() as robot:
    robot.home(blocking=True)

    stream = robot.start_streaming(speed=0.4, drop_last=DROP_LAST)

    for step in range(100):
        obs = robot.get_joints()
        waypoints = policy.predict_horizon(obs)  # (16, 7) array

        committed_end = (step + 1) * (HORIZON - DROP_LAST)
        while stream.last_completed_index + 1 < committed_end:
            time.sleep(0.01)

        stream.add_waypoints(waypoints)

    robot.stop()
```

## Provenance and License

`fr3` (version 0.1.0) is an internal lab control library for the FR3. It was vendored into this repository on 2026-10-20 from an uncommitted working copy; it has no published upstream repository or release history. Only `src/fr3/`, `pyproject.toml` and this README were vendored; the upstream unit tests, examples and Docker/simulation-server harness were not. `LICENSE-libfranka` and `NOTICE-libfranka` were added for this release. Code behaviour is unchanged; only comments, docstrings and this README were edited: example IP addresses were replaced by placeholders, or by `AURA_ROBOT_IP` in this README's examples; comments and docstrings that disagreed with the code were corrected, as were this README's home pose, `set_velocity` signature and Speed Formats section; the `move_to_pose` examples in this README and the `FR3` docstring now keep the hand pointing down (roll = π instead of 0); a comment naming a private simulator was removed; a libfranka attribution header (Apache-2.0) was added to `src/fr3/_motion.py`; and this README gained a Requirements section, a rewritten Installation section and notes on the smoother defaults and the orientation at `HOME_Q`, and lost its "Testing with Sim Server" section.

**License:** to be confirmed by the authors. None of the files carried a license or copyright notice when they were vendored. `JointMotionGenerator` in `src/fr3/_motion.py` is derived from `MotionGenerator` in `pylibfranka/examples/example_common.py` of [libfranka](https://github.com/frankarobotics/libfranka) (Copyright 2023 Franka Robotics GmbH, Apache License 2.0). [LICENSE-libfranka](LICENSE-libfranka) and [NOTICE-libfranka](NOTICE-libfranka) in this directory are verbatim copies of libfranka 0.20.5's `LICENSE` and `NOTICE` files. The comment header of `src/fr3/_motion.py` carries this attribution and lists the fr3 authors' changes to the libfranka code. Whatever license is chosen for `fr3`, the libfranka-derived code stays under the Apache License 2.0, so that header and the two files must be kept.
