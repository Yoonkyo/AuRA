"""Gamepad teleop controller for FR3.

Provides the ``Gamepad`` class that handles all robot-side filtering and
safety for gamepad teleoperation, and the ``deadzone`` utility function
for processing joystick axis values.
"""

from __future__ import annotations

import fcntl
import math
import os
import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fr3.robot import FR3

# ---------------------------------------------------------------------------
# Teleop collision thresholds — relaxed for dynamic teleoperation.
# Default robot thresholds [20, 40, 10, 20] are too tight for fast
# direction changes; the robot's own inertial forces trigger false reflexes.
# ---------------------------------------------------------------------------
_TELEOP_TORQUE_LOWER = [30.0] * 7
_TELEOP_TORQUE_UPPER = [60.0] * 7
_TELEOP_FORCE_LOWER = [15.0] * 6
_TELEOP_FORCE_UPPER = [30.0] * 6
_DEFAULT_GRIPPER_OPEN_WIDTH = 0.08
_DEFAULT_GRIPPER_SPEED = 0.1
_DEFAULT_GRIPPER_GRASP_WIDTH = 0.0
_DEFAULT_GRIPPER_GRASP_FORCE = 20.0
_DEFAULT_GRIPPER_GRASP_EPSILON = 0.08


def deadzone(value: float, threshold: float = 0.1) -> float:
    """Apply quadratic deadzone: zero below threshold, smooth ramp above.

    Returns a value in [-1, 1] with zero derivative at the threshold
    boundary, eliminating twitch from stick noise.

    Args:
        value: Raw axis value in [-1, 1].
        threshold: Deadzone radius (default 0.1).

    Returns:
        Filtered value in [-1, 1] with smooth onset.
    """
    if abs(value) < threshold:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    normalized = (abs(value) - threshold) / (1.0 - threshold)
    return sign * normalized * normalized  # quadratic — zero derivative at edge


class Gamepad:
    """Gamepad teleop controller — handles all robot-side filtering and safety.

    The caller handles controller input (e.g. pygame) and passes velocity
    commands via :meth:`update`.  This class handles:

      - EMA smoothing on input
      - Input-side acceleration limiting
      - Velocity magnitude clamping
      - Collision threshold management
      - Reflex error recovery with diagnostics
      - Status display (throttled, non-blocking)

    Typical usage::

        teleop = fr3.Gamepad(robot, max_linear_vel=0.12, max_angular_vel=0.4)
        teleop.start()
        while running:
            teleop.update(vx, vy, vz, wx, wy, wz)
            time.sleep(0.01)
        teleop.stop()

    Args:
        robot: Connected FR3 instance.
        max_linear_vel: Max translation speed in m/s (default 0.12).
        max_angular_vel: Max rotation speed in rad/s (default 0.4).
        ema_alpha: Smoothing factor 0-1, lower = smoother (default 0.3).
        max_linear_accel: Input acceleration limit in m/s² (default 4.0).
        max_angular_accel: Input acceleration limit in rad/s² (default 10.0).
        command_timeout: Seconds without a fresh input command before the 1 kHz
            velocity loop ramps to zero (default 0.5).
        gripper_open_width: Release width in meters when toggling open.
        gripper_open_speed: Opening speed in m/s.
        gripper_grasp_width: Target grasp width in meters when toggling closed.
        gripper_grasp_speed: Grasp speed in m/s.
        gripper_grasp_force: Grasping force in N.
        gripper_grasp_epsilon_inner: Allowed undershoot below grasp width in meters.
        gripper_grasp_epsilon_outer: Allowed overshoot above grasp width in meters.
    """

    def __init__(
        self,
        robot: FR3,
        max_linear_vel: float = 0.12,
        max_angular_vel: float = 0.4,
        ema_alpha: float = 0.3,
        max_linear_accel: float = 4.0,
        max_angular_accel: float = 10.0,
        command_timeout: float = 0.5,
        gripper_open_width: float = _DEFAULT_GRIPPER_OPEN_WIDTH,
        gripper_open_speed: float = _DEFAULT_GRIPPER_SPEED,
        gripper_grasp_width: float = _DEFAULT_GRIPPER_GRASP_WIDTH,
        gripper_grasp_speed: float = _DEFAULT_GRIPPER_SPEED,
        gripper_grasp_force: float = _DEFAULT_GRIPPER_GRASP_FORCE,
        gripper_grasp_epsilon_inner: float = _DEFAULT_GRIPPER_GRASP_EPSILON,
        gripper_grasp_epsilon_outer: float = _DEFAULT_GRIPPER_GRASP_EPSILON,
    ) -> None:
        self._robot = robot
        self._max_linear_vel = max_linear_vel
        self._max_angular_vel = max_angular_vel
        self._ema_alpha = ema_alpha
        self._max_accel = [max_linear_accel] * 3 + [max_angular_accel] * 3
        self._smoother_params = {"command_timeout": command_timeout}

        # Filter state
        self._filtered = [0.0] * 6
        self._prev_cmd = [0.0] * 6
        self._last_update = time.monotonic()

        # Gripper state
        self._gripper_open = True
        self._gripper_open_width = gripper_open_width
        self._gripper_open_speed = gripper_open_speed
        self._gripper_grasp_kwargs = {
            "width": gripper_grasp_width,
            "speed": gripper_grasp_speed,
            "force": gripper_grasp_force,
            "epsilon_inner": gripper_grasp_epsilon_inner,
            "epsilon_outer": gripper_grasp_epsilon_outer,
        }
        self._gripper_busy = False
        self._gripper_lock = threading.Lock()
        self._ignore_inputs_until = 0.0

        # Status display
        self._tick = 0
        self._status_every = 3  # at 100 Hz → ~33 Hz status updates
        self._stdout_fd: int | None = None
        self._stdout_fl_orig: int | None = None

        self._running = False
        self._recovery_until = 0.0  # monotonic time — skip set_velocity until then

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Set teleop collision thresholds and prepare for velocity control.

        Call once before the update loop.  Non-blocking.
        """
        self._robot.set_collision_thresholds(
            torque_lower=_TELEOP_TORQUE_LOWER,
            torque_upper=_TELEOP_TORQUE_UPPER,
            force_lower=_TELEOP_FORCE_LOWER,
            force_upper=_TELEOP_FORCE_UPPER,
        )

        # Make stdout non-blocking so status display can never stall the loop.
        try:
            self._stdout_fd = sys.stdout.fileno()
            self._stdout_fl_orig = fcntl.fcntl(self._stdout_fd, fcntl.F_GETFL)
            fcntl.fcntl(
                self._stdout_fd,
                fcntl.F_SETFL,
                self._stdout_fl_orig | os.O_NONBLOCK,
            )
        except (OSError, ValueError):
            self._stdout_fd = None

        self._last_update = time.monotonic()
        self._running = True
        self._start_velocity_control()
        print("Gamepad teleop active.")

    def update(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        wx: float = 0.0,
        wy: float = 0.0,
        wz: float = 0.0,
    ) -> None:
        """Filter input and send velocity command to the robot.

        Call at ~100 Hz from your input loop.  Handles reflex recovery
        internally — the caller does not need to catch errors.

        Args:
            vx, vy, vz: Linear velocity in m/s (robot base frame).
            wx, wy, wz: Angular velocity in rad/s (robot base frame).
        """
        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now
        # Clamp dt to avoid huge jumps after pauses (recovery, homing, etc.)
        dt = min(dt, 0.1)

        if self._gripper_busy or now < self._ignore_inputs_until:
            raw = [0.0] * 6
        else:
            raw = [vx, vy, vz, wx, wy, wz]

        # ── 1. Adaptive EMA filter ──────────────────────────────────
        # Use faster alpha when decaying toward zero (stick release) for
        # responsive stopping.  Normal alpha on ramp-up for smoothness.
        # The 1kHz jerk-limited smoother handles motion smoothing, so the
        # input-side EMA only needs to filter gamepad noise.
        for i in range(6):
            if abs(raw[i]) < abs(self._filtered[i]):
                # Decaying — use 2.5× alpha (capped at 0.7) for fast release
                alpha = min(self._ema_alpha * 2.5, 0.7)
            else:
                alpha = self._ema_alpha
            self._filtered[i] = alpha * raw[i] + (1.0 - alpha) * self._filtered[i]

        # Snap to zero when input is zero and residual is negligible.
        # The deadzone guarantees zero input for released sticks, so
        # this eliminates the long EMA tail without affecting active input.
        if all(abs(r) < 1e-10 for r in raw):
            for i in range(6):
                if abs(self._filtered[i]) < 5e-4:
                    self._filtered[i] = 0.0

        cmd = list(self._filtered)

        # ── 2. Input acceleration limiting ────────────────────────────
        if dt > 0:
            for i in range(6):
                delta = cmd[i] - self._prev_cmd[i]
                max_delta = self._max_accel[i] * dt
                if abs(delta) > max_delta:
                    cmd[i] = self._prev_cmd[i] + max_delta * (
                        1.0 if delta > 0 else -1.0
                    )

        # ── 3. Velocity magnitude clamping ────────────────────────────
        lin_mag = math.sqrt(cmd[0] ** 2 + cmd[1] ** 2 + cmd[2] ** 2)
        if lin_mag > self._max_linear_vel:
            scale = self._max_linear_vel / lin_mag
            cmd[0] *= scale
            cmd[1] *= scale
            cmd[2] *= scale

        ang_mag = math.sqrt(cmd[3] ** 2 + cmd[4] ** 2 + cmd[5] ** 2)
        if ang_mag > self._max_angular_vel:
            scale = self._max_angular_vel / ang_mag
            cmd[3] *= scale
            cmd[4] *= scale
            cmd[5] *= scale

        self._prev_cmd = list(cmd)

        # ── 4. Send to robot (non-blocking) ──────────────────────────
        # During recovery cooldown, skip set_velocity to avoid blocking
        # on thread restart.  The velocity loop does its own error
        # recovery (automatic_error_recovery) when it restarts.
        if self._recovery_until > now:
            return

        try:
            self._robot.set_velocity(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5])
        except RuntimeError as e:
            self._write_nonblocking(
                f"\r  [reflex error: {e}]" + " " * 20
            )
            self._reset_filters()
            self._recovery_until = now + 0.2  # 200ms cooldown
            return

        # ── 5. Throttled status display ───────────────────────────────
        self._tick += 1
        if self._tick % self._status_every == 0:
            self._show_status(cmd)

    def home(self) -> None:
        """Stop motion, home the robot, and resume velocity control.

        Blocking — waits for the home motion to complete, then restarts
        velocity control with a zero command.
        """
        print("\nHoming...")
        try:
            self._robot.stop()
        except Exception:
            pass
        self._robot.home(blocking=True)
        self._reset_filters()
        self._recovery_until = 0.0  # clear cooldown
        self._start_velocity_control()
        print("Homed. Resuming teleop.")

    def toggle_gripper(self) -> None:
        """Toggle gripper between release and grasp (non-blocking).

        The gripper operation runs in a background thread. Some pylibfranka
        gripper calls hold the Python GIL while blocking, so velocity commands
        are forced to zero before and immediately after the action. This avoids
        replaying stale joystick input when Python scheduling resumes.
        """
        with self._gripper_lock:
            if self._gripper_busy:
                return
            self._gripper_busy = True

        self._gripper_open = not self._gripper_open
        if self._gripper_open:
            threading.Thread(
                target=self._run_gripper_action,
                kwargs={
                    "target": self._robot.move_gripper,
                    "kwargs": {
                        "width": self._gripper_open_width,
                        "speed": self._gripper_open_speed,
                    },
                },
                daemon=True,
            ).start()
            return

        threading.Thread(
            target=self._run_gripper_action,
            kwargs={
                "target": self._robot.grasp,
                "kwargs": self._gripper_grasp_kwargs,
            },
            daemon=True,
        ).start()

    def stop(self) -> None:
        """Ramp down velocity, restore collision thresholds, and stop motion.

        Call once when done with teleop.
        """
        if not self._running:
            return

        # Send zero velocity to trigger smooth ramp-down in the 1 kHz smoother.
        try:
            self._robot.set_velocity(0, 0, 0, 0, 0, 0)
            time.sleep(0.1)  # brief ramp-down
        except Exception:
            pass

        try:
            self._robot.stop()
        except Exception:
            pass

        # Restore default collision thresholds.
        try:
            self._robot.set_collision_thresholds()
        except Exception:
            pass

        # Restore blocking stdout.
        if self._stdout_fd is not None and self._stdout_fl_orig is not None:
            try:
                fcntl.fcntl(self._stdout_fd, fcntl.F_SETFL, self._stdout_fl_orig)
            except OSError:
                pass

        self._running = False
        print("\nTeleop stopped.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_filters(self) -> None:
        """Zero all filter state for a clean restart."""
        self._filtered = [0.0] * 6
        self._prev_cmd = [0.0] * 6
        self._last_update = time.monotonic()

    def _zero_velocity(self) -> None:
        """Reset filter state and command a zero velocity."""
        self._reset_filters()
        try:
            self._robot.set_velocity(
                0, 0, 0, 0, 0, 0, smoother_params=self._smoother_params
            )
        except Exception:
            pass

    def _start_velocity_control(self) -> None:
        """Start or refresh the teleop velocity stream with a zero command."""
        self._robot.set_velocity(
            0, 0, 0, 0, 0, 0, smoother_params=self._smoother_params
        )

    def _run_gripper_action(self, target, kwargs: dict) -> None:
        """Run a blocking gripper call without leaving stale teleop command state."""
        self._zero_velocity()
        try:
            target(**kwargs)
        except Exception as exc:
            self._write_nonblocking(f"\r  [gripper error: {exc}]" + " " * 20)
        finally:
            self._zero_velocity()
            self._ignore_inputs_until = time.monotonic() + 0.15
            with self._gripper_lock:
                self._gripper_busy = False

    def _show_status(self, cmd: list[float]) -> None:
        """Print velocity + pose status line (non-blocking, throttled).

        Reads position directly from the control thread's cached state
        to guarantee this never blocks (avoids get_pose → read_once path).
        """
        try:
            ctrl = getattr(self._robot, "_control", None)
            if ctrl is None:
                return
            state = ctrl.get_latest_state()
            if state is None:
                return  # no state available — skip silently
            # Extract position from column-major 4×4 transform
            ee = state.O_T_EE
            px, py, pz = ee[12], ee[13], ee[14]
            line = (
                f"\r  vel ({cmd[0]:+.3f} {cmd[1]:+.3f} {cmd[2]:+.3f}) "
                f"rot ({cmd[3]:+.3f} {cmd[4]:+.3f} {cmd[5]:+.3f}) "
                f"pos ({px:.3f} {py:.3f} {pz:.3f})"
            )
        except Exception:
            return

        self._write_nonblocking(line)

    def _write_nonblocking(self, msg: str) -> None:
        """Write to stdout without blocking (drop if terminal can't keep up)."""
        if self._stdout_fd is not None:
            try:
                os.write(self._stdout_fd, msg.encode())
            except BlockingIOError:
                pass  # terminal can't keep up — drop this frame
        else:
            try:
                print(msg, end="", flush=True)
            except OSError:
                pass
