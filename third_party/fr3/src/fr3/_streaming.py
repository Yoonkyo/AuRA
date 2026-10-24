"""Streaming joint waypoint interpolator for policy deployment.

Provides smooth, continuous 1kHz joint position commands from a stream of
waypoints that arrive in asynchronous batches.  Uses quintic polynomial
interpolation with Catmull-Rom velocity blending to guarantee:

  - Every waypoint is hit exactly (position continuity).
  - Matching velocity and zero acceleration where segments meet at a waypoint
    (time past a segment's end is dropped, so velocity can dip for one tick).
  - Deceleration to zero velocity at the terminal waypoint, except that after
    ``finish()`` with ``drop_last`` the velocity can drop to zero in one tick.
  - Resumption without a velocity jump when new waypoints arrive before (or
    after) stopping; before, the rest of the segment toward the terminal
    waypoint is re-planned from zero acceleration (an acceleration jump).
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from fr3._types import validate_joint_limits

_VEL_MAX = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])


def _quintic_coeffs(
    q0: np.ndarray,
    q1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    T: float,
) -> np.ndarray:
    """Compute quintic polynomial coefficients for all 7 joints.

    The polynomial ``q(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵``
    satisfies boundary conditions with zero acceleration at both ends:

        q(0) = q0,   q(T) = q1
        q'(0) = v0,  q'(T) = v1
        q''(0) = 0,  q''(T) = 0

    Returns:
        (7, 6) array — ``coeffs[j]`` = ``[a0, a1, a2, a3, a4, a5]`` for joint *j*.
    """
    if T < 1e-10:
        return np.column_stack([q1, np.zeros((7, 5))])

    h = q1 - q0
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5

    a0 = q0
    a1 = v0
    a2 = np.zeros(7)
    a3 = (10.0 * h - (6.0 * v0 + 4.0 * v1) * T) / T3
    a4 = (-15.0 * h + (8.0 * v0 + 7.0 * v1) * T) / T4
    a5 = (6.0 * h - 3.0 * (v0 + v1) * T) / T5

    return np.column_stack([a0, a1, a2, a3, a4, a5])


def _eval_quintic(coeffs: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate position and velocity of a quintic polynomial at time *t*.

    Args:
        coeffs: (7, 6) coefficient array from :func:`_quintic_coeffs`.
        t: Time within the segment (seconds).

    Returns:
        ``(q, v)`` — position (7,) and velocity (7,) at time *t*.
    """
    t2, t3, t4, t5 = t**2, t**3, t**4, t**5
    t_pow = np.array([1.0, t, t2, t3, t4, t5])
    dt_pow = np.array([0.0, 1.0, 2.0 * t, 3.0 * t2, 4.0 * t3, 5.0 * t4])
    q = coeffs @ t_pow
    v = coeffs @ dt_pow
    return q, v


class StreamingMotionGenerator:
    """Generates smooth 1 kHz joint commands from an asynchronous waypoint stream.

    Waypoints can be added at any time via :meth:`add_waypoints`.  The
    generator interpolates between consecutive waypoints using quintic
    polynomials with Catmull-Rom velocity blending at interior waypoints and
    zero-velocity boundary conditions at the terminal waypoint.

    When the last enqueued waypoint is reached, the robot decelerates to a
    stop and holds position.  If new waypoints arrive before (or after)
    stopping, motion resumes seamlessly.

    Thread safety
    -------------
    All public methods are safe to call from any thread.  Internally a single
    lock serialises access to the waypoint queue and segment state.

    Parameters
    ----------
    speed_factor : float
        Fraction of maximum joint velocity (0–1).  Controls the per-segment
        duration: ``T = max(max_j(|Δq_j| / (v_max_j * speed_factor)) * 1.5, 0.1)``.
    drop_last : int
        Number of speculative waypoints at the tail of the queue that are
        replaced when :meth:`add_waypoints` is called.  When ``drop_last=0``
        (default), new waypoints are always appended.  When ``drop_last=N``,
        up to *N* waypoints are removed from the end of the queue before
        the new batch is appended, implementing an MPC-style rolling horizon.
    """

    def __init__(self, speed_factor: float = 0.3, drop_last: int = 0) -> None:
        self._lock = threading.Lock()
        self._waypoints: deque[np.ndarray] = deque()
        self._speed_factor = float(np.clip(speed_factor, 0.01, 1.0))
        self._drop_last = max(int(drop_last), 0)

        self._seg_q0 = np.zeros(7)
        self._seg_q1 = np.zeros(7)
        self._seg_v0 = np.zeros(7)
        self._seg_v1 = np.zeros(7)
        self._seg_T: float = 0.0
        self._seg_t: float = 0.0
        self._coeffs = np.zeros((7, 6))

        self._initialized = False
        self._idle = True
        self._last_q = np.zeros(7)
        self._last_v = np.zeros(7)

        self._completed_count: int = 0
        self._last_completed_wp: np.ndarray | None = None

    @property
    def waypoints_remaining(self) -> int:
        """Number of waypoints not yet fully reached.

        Includes the current segment target (if the robot is still en-route)
        plus all queued waypoints.
        """
        with self._lock:
            n = len(self._waypoints)
            if not self._idle:
                n += 1
            return n

    @property
    def last_completed_waypoint(self) -> np.ndarray | None:
        """Joint angles of the most recently completed waypoint, or ``None``.

        A waypoint is "completed" when the quintic segment targeting it
        finishes and the generator advances to the next segment (or goes idle).
        """
        with self._lock:
            if self._last_completed_wp is None:
                return None
            return self._last_completed_wp.copy()

    @property
    def last_completed_index(self) -> int:
        """Zero-based index of the most recently completed waypoint.

        Returns -1 if no waypoint has been completed yet.  The index counts
        every waypoint that was actually *reached*, regardless of whether
        later waypoints were dropped via ``drop_last``.
        """
        with self._lock:
            return self._completed_count - 1

    def error_to_waypoint(self, index: int) -> np.ndarray:
        """Per-joint signed error between the current position and a remaining waypoint.

        Index 0 is the next waypoint to be reached (current segment target if
        moving, otherwise the first queued waypoint).  Returns shape (7,).

        Raises:
            IndexError: If *index* is out of range.
        """
        with self._lock:
            remaining: list[np.ndarray] = []
            if not self._idle:
                remaining.append(self._seg_q1)
            remaining.extend(self._waypoints)
            if index < 0 or index >= len(remaining):
                raise IndexError(
                    f"index {index} out of range for {len(remaining)} remaining waypoints"
                )
            return remaining[index] - self._last_q

    def is_close(self, index: int, tolerance: float) -> bool:
        """True if every joint is within *tolerance* (rad) of the waypoint at *index*.

        Equivalent to ``max(abs(error_to_waypoint(index))) <= tolerance``.
        Returns False if *index* is out of range (no remaining waypoints).
        """
        try:
            err = self.error_to_waypoint(index)
        except IndexError:
            return False
        return bool(np.max(np.abs(err)) <= tolerance)

    def finish(self, run_remaining: bool = False) -> None:
        """Signal that no more batches will be sent.

        By default, drops the speculative tail (up to ``drop_last`` waypoints)
        so the robot stops after the last committed waypoint.  Pass
        ``run_remaining=True`` to keep and execute all queued waypoints.
        """
        if run_remaining or self._drop_last == 0:
            return
        with self._lock:
            to_drop = min(self._drop_last, len(self._waypoints))
            for _ in range(to_drop):
                self._waypoints.pop()

    def add_waypoints(self, waypoints: np.ndarray) -> None:
        """Enqueue one or more joint-angle waypoints.

        If ``drop_last > 0`` was set at construction, up to that many
        waypoints are removed from the tail of the queue before the new
        batch is appended.

        Args:
            waypoints: Shape ``(7,)`` for a single waypoint or ``(N, 7)`` for
                a batch.  Each row is a full 7-joint configuration in radians.
        """
        waypoints = np.atleast_2d(np.asarray(waypoints, dtype=float))
        if waypoints.ndim != 2 or waypoints.shape[1] != 7:
            raise ValueError(
                f"waypoints must be (N, 7) or (7,), got shape {waypoints.shape}"
            )
        validate_joint_limits(waypoints)
        with self._lock:
            if self._drop_last > 0:
                to_drop = min(self._drop_last, len(self._waypoints))
                for _ in range(to_drop):
                    self._waypoints.pop()
            for wp in waypoints:
                self._waypoints.append(wp.copy())
            if not self._idle:
                self._maybe_recompute_terminal()

    def step(self, current_q: np.ndarray, dt: float) -> tuple[np.ndarray, bool]:
        """Advance one control-loop timestep.

        Args:
            current_q: Actual joint angles from ``RobotState.q`` (used only on
                the very first call to seed the starting position).
            dt: Time since last call (seconds), typically 0.001.

        Returns:
            ``(target_q, finished)`` — *finished* is always ``False``; the
            streaming loop is terminated externally via ``ControlThread.stop()``.
        """
        with self._lock:
            if not self._initialized:
                self._last_q = np.array(current_q, dtype=float)
                self._initialized = True

            if self._idle:
                if not self._waypoints:
                    return self._last_q.copy(), False
                self._begin_next_segment()

            self._seg_t += dt

            if self._seg_t >= self._seg_T:
                self._last_q = self._seg_q1.copy()
                self._last_v = self._seg_v1.copy()
                self._last_completed_wp = self._seg_q1.copy()
                self._completed_count += 1

                if self._waypoints:
                    self._begin_next_segment()
                else:
                    self._idle = True
                    self._last_v = np.zeros(7)
            else:
                self._last_q, self._last_v = _eval_quintic(self._coeffs, self._seg_t)

            return self._last_q.copy(), False

    def _begin_next_segment(self) -> None:
        """Pop the next waypoint and compute a quintic segment toward it."""
        self._seg_q0 = self._last_q.copy()
        self._seg_v0 = self._last_v.copy()
        self._seg_q1 = self._waypoints.popleft()

        self._seg_T = self._compute_duration(self._seg_q0, self._seg_q1)

        if self._waypoints:
            next_wp = self._waypoints[0]
            next_T = self._compute_duration(self._seg_q1, next_wp)
            self._seg_v1 = self._catmull_rom_vel(
                self._seg_q0, self._seg_q1, next_wp, self._seg_T, next_T
            )
        else:
            self._seg_v1 = np.zeros(7)

        self._coeffs = _quintic_coeffs(
            self._seg_q0,
            self._seg_q1,
            self._seg_v0,
            self._seg_v1,
            self._seg_T,
        )
        self._seg_t = 0.0
        self._idle = False

    def _maybe_recompute_terminal(self) -> None:
        """If decelerating toward a terminal waypoint and new waypoints have
        arrived, recompute the remaining trajectory with a nonzero exit
        velocity so the robot does not slow down unnecessarily.
        """
        if not np.allclose(self._seg_v1, 0.0):
            return
        if not self._waypoints:
            return

        remaining = self._seg_T - self._seg_t
        if remaining < 0.02:
            return

        q_now, v_now = _eval_quintic(self._coeffs, self._seg_t)

        next_wp = self._waypoints[0]
        next_T = self._compute_duration(self._seg_q1, next_wp)
        new_v1 = self._catmull_rom_vel(q_now, self._seg_q1, next_wp, remaining, next_T)

        self._seg_q0 = q_now.copy()
        self._seg_v0 = v_now.copy()
        self._seg_v1 = new_v1
        self._seg_T = remaining
        self._seg_t = 0.0
        self._coeffs = _quintic_coeffs(
            self._seg_q0,
            self._seg_q1,
            self._seg_v0,
            self._seg_v1,
            self._seg_T,
        )

    def _compute_duration(self, q_from: np.ndarray, q_to: np.ndarray) -> float:
        """Time-optimal duration for a segment, scaled by speed factor.

        ``T = max(max_j(|Δq_j| / (v_max_j * speed_factor)) * 1.5, 0.1)``; a
        rest-to-rest segment can still peak at 1.25 × v_max_j × speed_factor.
        """
        max_vel = _VEL_MAX * self._speed_factor
        disp = np.abs(q_to - q_from)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_joint = np.where(disp > 1e-8, disp / max_vel, 0.0)
        return float(max(np.max(per_joint) * 1.5, 0.1))

    @staticmethod
    def _catmull_rom_vel(
        q_prev: np.ndarray,
        q_curr: np.ndarray,
        q_next: np.ndarray,
        T_prev: float,
        T_next: float,
    ) -> np.ndarray:
        """Catmull-Rom tangent at *q_curr*, clamped to joint velocity limits."""
        v = (q_next - q_prev) / (T_prev + T_next)
        max_vel = _VEL_MAX * 0.8
        return np.clip(v, -max_vel, max_vel)
