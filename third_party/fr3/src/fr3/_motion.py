# JointMotionGenerator is derived from MotionGenerator in libfranka's
# pylibfranka/examples/example_common.py (Franka Robotics' Python port of
# examples/examples_common.{h,cpp}), https://github.com/frankarobotics/libfranka.
# libfranka: Copyright 2023 Franka Robotics GmbH. Licensed under the Apache License,
# Version 2.0; see third_party/fr3/LICENSE-libfranka and NOTICE-libfranka (copies of
# LICENSE and NOTICE at https://github.com/frankarobotics/libfranka/tree/0.20.5).
# Modified by the fr3 authors:
#   - renamed from MotionGenerator;
#   - constructor signature changed from (speed_factor, q_goal) to
#     (q_goal, speed_factor=0.3);
#   - q_goal and q_start stored as float arrays;
#   - __call__(robot_state, duration_sec) -> JointPositions replaced by
#     step(current_q, dt) -> (q, motion_finished);
#   - from_fraction, from_rad_s and from_duration constructors added;
#   - comments removed, type hints changed, code reformatted.
# Unchanged from upstream: the trajectory math, the velocity/acceleration
# limits and the 1e-12 motion-finished threshold.
# This notice covers JointMotionGenerator only.

"""Joint-space and Cartesian-space motion generators."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class JointMotionGenerator:
    def __init__(self, q_goal: np.ndarray, speed_factor: float = 0.3):
        self.q_goal = np.array(q_goal, dtype=float)
        self.time = 0.0

        self.dq_max = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]) * speed_factor
        self.ddq_max_start = (
            np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) * speed_factor
        )
        self.ddq_max_goal = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]) * speed_factor

        self.q_start = np.zeros(7)
        self.delta_q = np.zeros(7)
        self.dq_max_sync = np.zeros(7)
        self.t_1_sync = np.zeros(7)
        self.t_2_sync = np.zeros(7)
        self.t_f_sync = np.zeros(7)
        self.q_1 = np.zeros(7)

        self.delta_q_motion_finished = 1e-12
        self.initialized = False

    @classmethod
    def from_fraction(
        cls, q_goal: np.ndarray, fraction: float
    ) -> "JointMotionGenerator":
        return cls(q_goal=q_goal, speed_factor=fraction)

    @classmethod
    def from_rad_s(cls, q_goal: np.ndarray, max_vel_rad_s) -> "JointMotionGenerator":
        """Create generator from max velocity in rad/s.

        Args:
            q_goal: Target joint angles.
            max_vel_rad_s: Max velocity as scalar (applied to all joints) or
                7-element array for per-joint velocity limits.
        """
        max_vel = np.asarray(max_vel_rad_s, dtype=float)
        if max_vel.ndim == 0:
            # Scalar: uniform speed factor
            speed_factor = float(np.clip(max_vel / 2.5, 0.01, 1.0))
            return cls(q_goal=q_goal, speed_factor=speed_factor)
        else:
            # Per-joint: create generator with speed_factor=1.0 and override dq_max
            gen = cls(q_goal=q_goal, speed_factor=1.0)
            # Per-joint max velocities (max joint vel is 2.5 rad/s)
            per_joint_max = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])
            gen.dq_max = np.clip(max_vel, 0.0, per_joint_max)
            gen.ddq_max_start = np.array([5.0] * 7) * np.clip(max_vel / 2.5, 0.01, 1.0)
            gen.ddq_max_goal = gen.ddq_max_start.copy()
            return gen

    @classmethod
    def from_duration(
        cls,
        q_goal: np.ndarray,
        current_q: np.ndarray,
        duration_seconds: float,
    ) -> "JointMotionGenerator":
        delta_q = np.abs(
            np.array(q_goal, dtype=float) - np.array(current_q, dtype=float)
        )
        max_delta = float(np.max(delta_q)) if delta_q.size > 0 else 0.0
        if max_delta <= 1e-12:
            speed_factor = 1.0
        else:
            reference_duration = 1.5 * (max_delta / (np.pi / 2.0))
            if duration_seconds <= 1e-12:
                speed_factor = 1.0
            else:
                speed_factor = reference_duration / duration_seconds
        speed_factor = float(np.clip(speed_factor, 0.01, 1.0))
        return cls(q_goal=q_goal, speed_factor=speed_factor)

    def calculate_synchronized_values(self) -> None:
        dq_max_reach = self.dq_max.copy()
        t_f = np.zeros(7)
        delta_t_2 = np.zeros(7)
        t_1 = np.zeros(7)
        delta_t_2_sync = np.zeros(7)
        sign_delta_q = np.sign(self.delta_q).astype(int)

        for i in range(7):
            if abs(self.delta_q[i]) > self.delta_q_motion_finished:
                if abs(self.delta_q[i]) < (
                    3.0 / 4.0 * (self.dq_max[i] ** 2 / self.ddq_max_start[i])
                    + 3.0 / 4.0 * (self.dq_max[i] ** 2 / self.ddq_max_goal[i])
                ):
                    dq_max_reach[i] = np.sqrt(
                        4.0
                        / 3.0
                        * self.delta_q[i]
                        * sign_delta_q[i]
                        * (self.ddq_max_start[i] * self.ddq_max_goal[i])
                        / (self.ddq_max_start[i] + self.ddq_max_goal[i])
                    )

                t_1[i] = 1.5 * dq_max_reach[i] / self.ddq_max_start[i]
                delta_t_2[i] = 1.5 * dq_max_reach[i] / self.ddq_max_goal[i]
                t_f[i] = (
                    t_1[i] / 2.0
                    + delta_t_2[i] / 2.0
                    + abs(self.delta_q[i]) / dq_max_reach[i]
                )

        max_t_f = np.max(t_f)

        for i in range(7):
            if abs(self.delta_q[i]) > self.delta_q_motion_finished:
                a = 1.5 / 2.0 * (self.ddq_max_goal[i] + self.ddq_max_start[i])
                b = -1.0 * max_t_f * self.ddq_max_goal[i] * self.ddq_max_start[i]
                c = abs(self.delta_q[i]) * self.ddq_max_goal[i] * self.ddq_max_start[i]

                delta = b * b - 4.0 * a * c
                if delta < 0.0:
                    delta = 0.0

                self.dq_max_sync[i] = (-1.0 * b - np.sqrt(delta)) / (2.0 * a)
                self.t_1_sync[i] = 1.5 * self.dq_max_sync[i] / self.ddq_max_start[i]
                delta_t_2_sync[i] = 1.5 * self.dq_max_sync[i] / self.ddq_max_goal[i]
                self.t_f_sync[i] = (
                    self.t_1_sync[i] / 2.0
                    + delta_t_2_sync[i] / 2.0
                    + abs(self.delta_q[i] / self.dq_max_sync[i])
                )
                self.t_2_sync[i] = self.t_f_sync[i] - delta_t_2_sync[i]
                self.q_1[i] = (
                    self.dq_max_sync[i] * sign_delta_q[i] * (0.5 * self.t_1_sync[i])
                )

    def calculate_desired_values(self, t: float) -> tuple[np.ndarray, bool]:
        delta_q_d = np.zeros(7)
        sign_delta_q = np.sign(self.delta_q).astype(int)
        t_d = self.t_2_sync - self.t_1_sync
        delta_t_2_sync = self.t_f_sync - self.t_2_sync
        joint_motion_finished = [False] * 7

        for i in range(7):
            if abs(self.delta_q[i]) < self.delta_q_motion_finished:
                delta_q_d[i] = 0
                joint_motion_finished[i] = True
            else:
                if t < self.t_1_sync[i]:
                    delta_q_d[i] = (
                        -1.0
                        / (self.t_1_sync[i] ** 3)
                        * self.dq_max_sync[i]
                        * sign_delta_q[i]
                        * (0.5 * t - self.t_1_sync[i])
                        * (t**3)
                    )
                elif t >= self.t_1_sync[i] and t < self.t_2_sync[i]:
                    delta_q_d[i] = (
                        self.q_1[i]
                        + (t - self.t_1_sync[i]) * self.dq_max_sync[i] * sign_delta_q[i]
                    )
                elif t >= self.t_2_sync[i] and t < self.t_f_sync[i]:
                    delta_q_d[i] = (
                        self.delta_q[i]
                        + 0.5
                        * (
                            1.0
                            / (delta_t_2_sync[i] ** 3)
                            * (t - self.t_1_sync[i] - 2.0 * delta_t_2_sync[i] - t_d[i])
                            * ((t - self.t_1_sync[i] - t_d[i]) ** 3)
                            + (
                                2.0 * t
                                - 2.0 * self.t_1_sync[i]
                                - delta_t_2_sync[i]
                                - 2.0 * t_d[i]
                            )
                        )
                        * self.dq_max_sync[i]
                        * sign_delta_q[i]
                    )
                else:
                    delta_q_d[i] = self.delta_q[i]
                    joint_motion_finished[i] = True

        return delta_q_d, all(joint_motion_finished)

    def step(self, current_q: np.ndarray, dt: float) -> tuple[np.ndarray, bool]:
        self.time += dt

        if not self.initialized:
            self.q_start = np.array(current_q, dtype=float)
            self.delta_q = self.q_goal - self.q_start
            self.calculate_synchronized_values()
            self.initialized = True

        delta_q_d, motion_finished = self.calculate_desired_values(self.time)
        return self.q_start + delta_q_d, motion_finished


class CartesianInterpolator:
    def __init__(self, start_pose: np.ndarray, end_pose: np.ndarray, duration: float):
        self.start_pose = np.array(start_pose, dtype=float)
        self.end_pose = np.array(end_pose, dtype=float)
        self.duration = float(duration)

        self.start_pos = self.start_pose[:3, 3]
        self.end_pos = self.end_pose[:3, 3]

        start_rot = Rotation.from_matrix(self.start_pose[:3, :3])
        end_rot = Rotation.from_matrix(self.end_pose[:3, :3])
        self.slerp = Slerp([0.0, 1.0], Rotation.concatenate([start_rot, end_rot]))

    def step(self, elapsed_time: float) -> tuple[np.ndarray, bool]:
        motion_finished = elapsed_time >= self.duration

        if self.duration <= 0.0:
            s = 1.0
        else:
            t = float(np.clip(elapsed_time, 0.0, self.duration))
            u = t / self.duration
            s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)

        pos = self.start_pos + s * (self.end_pos - self.start_pos)
        rot = self.slerp(s).as_matrix()

        out = np.eye(4, dtype=float)
        out[:3, :3] = rot
        out[:3, 3] = pos
        out[3, 3] = 1.0
        return out, motion_finished
