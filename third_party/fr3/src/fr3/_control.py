"""Background 1kHz control thread manager."""

from __future__ import annotations

import sys
import time
import threading
import numpy as np
from pylibfranka import (
    CartesianPose,
    CartesianVelocities,
    ControllerMode,
    JointPositions,
)

from fr3._motion import CartesianInterpolator, JointMotionGenerator
from fr3._streaming import StreamingMotionGenerator


# Hold final position for this many ticks (at 1 kHz) before sending
# motion_finished, giving the robot's internal controller time to settle.
_SETTLE_TICKS = 100  # 100 ms

# Cartesian motion retry parameters — on reflex errors, retry with longer
# duration to avoid joint acceleration discontinuities.
_MAX_CARTESIAN_RETRIES = 3
_DURATION_BACKOFF = 1.5


def _safe_stop(robot) -> None:
    """Call robot.stop() without raising — safe for finally blocks."""
    try:
        robot.stop()
    except Exception:
        pass


class ControlThread:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._state_lock = threading.Lock()
        self._latest_state = None
        self._error: Exception | None = None
        self._velocity_lock = threading.Lock()
        self._velocity_cmd = [0.0] * 6
        self._velocity_cmd_time = time.monotonic()

    def start_joint_motion(
        self,
        robot,
        motion_gen: JointMotionGenerator,
        controller_mode=ControllerMode.JointImpedance,
    ) -> None:
        self.stop()
        self._stop_event.clear()
        self._done_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._joint_motion_loop,
            args=(robot, motion_gen, controller_mode),
            daemon=True,
        )
        self._thread.start()

    def start_cartesian_motion(
        self,
        robot,
        target_4x4: np.ndarray,
        duration: float,
        controller_mode=ControllerMode.CartesianImpedance,
    ) -> None:
        self.stop()
        self._stop_event.clear()
        self._done_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._cartesian_motion_loop,
            args=(robot, target_4x4, duration, controller_mode),
            daemon=True,
        )
        self._thread.start()

    def start_streaming_motion(
        self,
        robot,
        streaming_gen: StreamingMotionGenerator,
        controller_mode=ControllerMode.JointImpedance,
    ) -> None:
        self.stop()
        self._stop_event.clear()
        self._done_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._streaming_motion_loop,
            args=(robot, streaming_gen, controller_mode),
            daemon=True,
        )
        self._thread.start()

    def start_velocity(
        self,
        robot,
        controller_mode=ControllerMode.CartesianImpedance,
        smoother_params: dict | None = None,
    ) -> None:
        self.stop()
        with self._velocity_lock:
            self._velocity_cmd = [0.0] * 6
            self._velocity_cmd_time = time.monotonic()
        self._stop_event.clear()
        self._done_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._velocity_loop,
            args=(robot, controller_mode, smoother_params),
            daemon=True,
        )
        self._thread.start()

    def set_velocity(self, linear_vel: list[float], angular_vel: list[float]) -> None:
        with self._velocity_lock:
            self._velocity_cmd = list(linear_vel) + list(angular_vel)
            self._velocity_cmd_time = time.monotonic()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        # Brief pause to let the robot's C++ side fully close the control
        # session after _safe_stop() runs in the thread's finally block.
        time.sleep(0.05)
        self._done_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> bool:
        result = self._done_event.wait(timeout=timeout)
        if self._error is not None:
            err = self._error
            self._error = None
            raise RuntimeError(f"Motion failed: {err}") from err
        return result

    def get_latest_state(self):
        with self._state_lock:
            return self._latest_state

    def get_error(self) -> Exception | None:
        return self._error

    def _joint_motion_loop(self, robot, motion_gen, controller_mode) -> None:
        try:
            robot.automatic_error_recovery()
            active_control = robot.start_joint_position_control(controller_mode)
            final_q = None
            settle_ticks = 0
            while not self._stop_event.is_set():
                state, duration = active_control.readOnce()
                with self._state_lock:
                    self._latest_state = state
                current_q = np.array(state.q)
                dt = duration.to_sec()

                if final_q is not None:
                    # Settling phase: hold final position until robot stops
                    settle_ticks += 1
                    cmd = JointPositions(list(final_q))
                    if settle_ticks >= _SETTLE_TICKS:
                        cmd.motion_finished = True
                        active_control.writeOnce(cmd)
                        break
                    active_control.writeOnce(cmd)
                    continue

                target_q, finished = motion_gen.step(current_q, dt)
                cmd = JointPositions(list(target_q))
                if finished:
                    final_q = target_q.copy()
                active_control.writeOnce(cmd)
        except Exception as exc:
            self._error = exc
        finally:
            _safe_stop(robot)
            with self._state_lock:
                self._latest_state = None
            self._done_event.set()

    def _cartesian_motion_loop(self, robot, target_4x4, duration, controller_mode) -> None:
        last_error = None

        for _attempt in range(1 + _MAX_CARTESIAN_RETRIES):
            try:
                robot.automatic_error_recovery()
                active_control = robot.start_cartesian_pose_control(controller_mode)
                elapsed = 0.0
                final_flat = None
                settle_ticks = 0
                interpolator = None
                while not self._stop_event.is_set():
                    state, dur = active_control.readOnce()
                    with self._state_lock:
                        self._latest_state = state
                    dt = dur.to_sec()

                    # Build interpolator from robot's actual pose on first tick
                    if interpolator is None:
                        start_4x4 = np.array(state.O_T_EE).reshape(4, 4, order="F")
                        interpolator = CartesianInterpolator(
                            start_4x4, target_4x4, duration
                        )

                    if final_flat is not None:
                        # Settling phase: hold final pose until robot stops
                        settle_ticks += 1
                        cmd = CartesianPose(final_flat)
                        if settle_ticks >= _SETTLE_TICKS:
                            cmd.motion_finished = True
                            active_control.writeOnce(cmd)
                            break
                        active_control.writeOnce(cmd)
                        continue

                    pose_4x4, finished = interpolator.step(elapsed)
                    elapsed += dt
                    flat = list(pose_4x4.flatten(order="F"))
                    cmd = CartesianPose(flat)
                    if finished:
                        final_flat = flat
                    active_control.writeOnce(cmd)

                # Success — exit the retry loop
                last_error = None
                break

            except Exception as exc:
                last_error = exc
                _safe_stop(robot)
                # Only retry on reflex-type errors
                exc_str = str(exc).lower()
                if "reflex" not in exc_str and "discontinuity" not in exc_str:
                    break
                # Increase duration for next attempt
                duration *= _DURATION_BACKOFF

        if last_error is not None:
            self._error = last_error

        _safe_stop(robot)
        with self._state_lock:
            self._latest_state = None
        self._done_event.set()

    def _streaming_motion_loop(self, robot, streaming_gen, controller_mode) -> None:
        try:
            robot.automatic_error_recovery()
            active_control = robot.start_joint_position_control(controller_mode)
            while not self._stop_event.is_set():
                state, duration = active_control.readOnce()
                with self._state_lock:
                    self._latest_state = state
                current_q = np.array(state.q)
                dt = duration.to_sec()
                target_q, _ = streaming_gen.step(current_q, dt)
                cmd = JointPositions(list(target_q))
                active_control.writeOnce(cmd)

            state, _duration = active_control.readOnce()
            cmd = JointPositions(list(state.q))
            cmd.motion_finished = True
            active_control.writeOnce(cmd)
        except Exception as exc:
            self._error = exc
        finally:
            _safe_stop(robot)
            with self._state_lock:
                self._latest_state = None
            self._done_event.set()

    def _velocity_loop(self, robot, controller_mode, smoother_params=None) -> None:
        # Pose-integrated velocity control (inspired by DS4 franka controller).
        #
        # Instead of commanding CartesianVelocities (which drifts during
        # rotation), we integrate the user's velocity into a target 4×4 pose
        # and command CartesianPose.  This guarantees:
        #   1. Zero positional drift during rotation-only control
        #   2. Smooth motion via jerk-limited velocity smoothing
        #   3. No velocity/acceleration discontinuities
        #
        # The jerk-limited smoother with preemptive braking ensures the
        # per-tick pose deltas change smoothly — no sudden jumps.
        params = smoother_params or {}
        DT = 0.001
        K = params.get("K", 15.0)  # proportional gain (controls convergence speed)
        SNAP_THRESHOLD = 1e-6   # snap to target when this close
        max_accel = np.array(params.get("max_accel", [6.0] * 3 + [12.0] * 3))     # m/s², rad/s²
        max_jerk = np.array(params.get("max_jerk", [0.04] * 3 + [0.08] * 3))      # per tick
        # This is a watchdog for the low-rate input producer, not the 1 kHz
        # libfranka stream.  Keep it comfortably above normal pygame/Python
        # scheduling jitter so brief UI stalls do not pulse the robot to stop.
        command_timeout = params.get("command_timeout", 0.5)

        try:
            robot.automatic_error_recovery()
            active_control = robot.start_cartesian_pose_control(controller_mode)
            smoothed = np.zeros(6)
            accel = np.zeros(6)
            target_4x4 = None  # initialized from first readOnce()
            tick = 0
            anchor_pos = None  # position anchor for rotation-only control

            while True:
                state, _duration = active_control.readOnce()
                with self._state_lock:
                    self._latest_state = state

                # Initialize target pose from first robot state
                if target_4x4 is None:
                    target_4x4 = np.array(state.O_T_EE).reshape(
                        4, 4, order="F"
                    ).copy()

                # Target velocity: zero when stopping, else user command
                if self._stop_event.is_set():
                    target = np.zeros(6)
                else:
                    with self._velocity_lock:
                        target = np.array(self._velocity_cmd)
                        cmd_age = time.monotonic() - self._velocity_cmd_time
                    if command_timeout is not None and cmd_age > command_timeout:
                        target = np.zeros(6)

                # Jerk-limited smoothing with preemptive braking.
                # Asymmetric: 3× jerk when decelerating for faster response
                # to direction changes and stop commands.
                for i in range(6):
                    err = target[i] - smoothed[i]

                    # Use higher jerk when decelerating (target closer to zero
                    # than current smoothed velocity) for faster stopping.
                    decelerating = abs(target[i]) < abs(smoothed[i])
                    jerk_i = max_jerk[i] * (3.0 if decelerating else 1.0)

                    # Velocity change needed to decelerate accel to zero
                    if abs(accel[i]) > 1e-10:
                        stop_vel_change = (
                            accel[i] * abs(accel[i]) / jerk_i * DT * 0.5
                        )
                    else:
                        stop_vel_change = 0.0

                    # Brake preemptively to avoid overshooting target
                    predicted = smoothed[i] + stop_vel_change
                    needs_brake = (accel[i] > 0 and predicted >= target[i]) or (
                        accel[i] < 0 and predicted <= target[i]
                    )

                    if needs_brake:
                        desired = 0.0
                    else:
                        desired = np.clip(K * err, -max_accel[i], max_accel[i])

                    da = np.clip(desired - accel[i], -jerk_i, jerk_i)
                    accel[i] += da
                    smoothed[i] += accel[i] * DT

                    # Snap to target when very close
                    if (
                        abs(smoothed[i] - target[i]) < SNAP_THRESHOLD
                        and abs(accel[i]) < SNAP_THRESHOLD
                    ):
                        smoothed[i] = target[i]
                        accel[i] = 0.0

                # --- Integrate smoothed velocity into target pose ---

                # Position: Euler integration with anchoring.
                # When target linear velocity is zero and the smoother has
                # settled, lock position to prevent accumulated numerical
                # drift during rotation-only control.
                target_lin_zero = np.max(np.abs(target[:3])) < 1e-10
                smoothed_lin_near_zero = np.max(np.abs(smoothed[:3])) < 1e-5

                if target_lin_zero and smoothed_lin_near_zero:
                    if anchor_pos is None:
                        anchor_pos = target_4x4[:3, 3].copy()
                    target_4x4[:3, 3] = anchor_pos.copy()
                else:
                    if not target_lin_zero:
                        anchor_pos = None  # release when translating
                    target_4x4[:3, 3] += smoothed[:3] * DT

                # Orientation: Rodrigues rotation in base frame
                omega = smoothed[3:]
                angle = np.linalg.norm(omega) * DT
                if angle > 1e-10:
                    axis = omega / np.linalg.norm(omega)
                    K_skew = np.array(
                        [
                            [0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0],
                        ]
                    )
                    dR = (
                        np.eye(3)
                        + np.sin(angle) * K_skew
                        + (1 - np.cos(angle)) * (K_skew @ K_skew)
                    )
                    # Pre-multiply: rotation in base frame
                    target_4x4[:3, :3] = dR @ target_4x4[:3, :3]

                # Re-orthogonalize rotation matrix every 500 ticks (0.5 s)
                # to prevent numerical drift during long teleop sessions.
                tick += 1
                if tick % 500 == 0:
                    U, _, Vt = np.linalg.svd(target_4x4[:3, :3])
                    target_4x4[:3, :3] = U @ Vt

                # Monitor communication health every second.
                if tick % 1000 == 0:
                    success_rate = getattr(
                        state, "control_command_success_rate", 1.0
                    )
                    if success_rate < 0.9:
                        print(
                            f"Warning: control_command_success_rate="
                            f"{success_rate:.2f}",
                            file=sys.stderr,
                        )

                flat = list(target_4x4.flatten(order="F"))
                cmd = CartesianPose(flat)

                # Finish only once fully ramped down to ~zero
                if self._stop_event.is_set() and np.max(np.abs(smoothed)) < 1e-6:
                    cmd.motion_finished = True
                    active_control.writeOnce(cmd)
                    break

                active_control.writeOnce(cmd)
        except Exception as exc:
            self._error = exc
        finally:
            _safe_stop(robot)
            with self._state_lock:
                self._latest_state = None
            self._done_event.set()
