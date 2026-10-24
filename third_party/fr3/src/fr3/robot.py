"""Main FR3 robot class exposing the public API."""

import importlib
import numpy as np
from pylibfranka import Robot

from fr3._control import ControlThread
from fr3._gripper import GripperControl
from fr3._kinematics import o_t_ee_to_pose, pose_to_o_t_ee
from fr3._motion import JointMotionGenerator
from fr3._streaming import StreamingMotionGenerator
from fr3._types import HOME_Q, Pose, validate_joint_limits


class FR3:
    """High-level control interface for the Franka Research 3 robot.

    Wraps pylibfranka to provide clean (x,y,z,roll,pitch,yaw) pose control,
    joint target tracking, velocity streaming, and gripper operations.

    Usage:
        import fr3
        fr3.ROBOT_IP = "<ROBOT_IP>"

        with fr3.FR3() as robot:
            robot.home(blocking=True)
            robot.move_to_pose(0.5, 0.0, 0.4, 3.14159, 0, 0, blocking=True)
            robot.open_gripper()
    """

    def __init__(self, ip: str | None = None):
        """Connect to robot at given IP address.

        Args:
            ip: Robot IP address. If None, uses fr3.ROBOT_IP module variable.

        Raises:
            ValueError: If no IP is configured.
        """
        if ip is None:
            ip = importlib.import_module("fr3").ROBOT_IP
        if ip is None:
            raise ValueError(
                "No robot IP configured. Set fr3.ROBOT_IP = '...' or pass ip to FR3(ip='...')"
            )

        self._ip = ip
        self._robot = Robot(ip)
        self._robot.automatic_error_recovery()
        self._robot.set_collision_behavior(
            [20.0] * 7,
            [40.0] * 7,
            [10.0] * 6,
            [20.0] * 6,
        )
        self._robot.set_joint_impedance(
            [3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0]
        )
        self._robot.set_cartesian_impedance(
            [3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0]
        )
        self._control = ControlThread()
        self._gripper = GripperControl(ip)
        self._streaming_gen: StreamingMotionGenerator | None = None
        self._smoother_params: dict | None = None

    def __enter__(self):
        """Enter context manager and return this robot instance."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and release all connections."""
        self.disconnect()
        return False

    def disconnect(self) -> None:
        """Stop motion and release TCP connections to the robot and gripper."""
        self.stop()
        robot = self.__dict__.pop("_robot", None)
        if robot is not None:
            del robot
        gripper_ctrl = self.__dict__.pop("_gripper", None)
        if gripper_ctrl is not None:
            inner = gripper_ctrl.__dict__.pop("_gripper", None)
            if inner is not None:
                del inner
            del gripper_ctrl

    def _get_state(self):
        """Get latest robot state from control thread, else read_once()."""
        state = self._control.get_latest_state()
        if state is None:
            state = self._robot.read_once()
        return state

    def get_pose(self) -> Pose:
        """Get current end-effector pose in robot base frame.

        Returns:
            Pose(x, y, z, roll, pitch, yaw) with meters and radians.
        """
        state = self._get_state()
        return o_t_ee_to_pose(state.O_T_EE)

    def get_joints(self) -> np.ndarray:
        """Get current joint angles in radians.

        Returns:
            7-element numpy array of joint angles [q1..q7] in radians.
        """
        state = self._get_state()
        return np.array(state.q)

    def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        speed: float = 0.3,
        blocking: bool = False,
    ) -> None:
        """Move end-effector to target pose using Cartesian interpolation.

        Args:
            x, y, z: Target position in meters (robot base frame).
            roll, pitch, yaw: Target ZYX Euler orientation in radians.
            speed: Speed fraction 0-1 (default 0.3). Controls duration.
            blocking: If True, block until motion completes.
        """
        target_pose = Pose(x, y, z, roll, pitch, yaw)
        target_flat = pose_to_o_t_ee(target_pose)
        target_4x4 = np.array(target_flat).reshape((4, 4), order="F")

        current_state = self._get_state()
        current_flat = current_state.O_T_EE
        current_4x4 = np.array(current_flat).reshape((4, 4), order="F")

        # Position distance
        pos_dist = np.linalg.norm(target_4x4[:3, 3] - current_4x4[:3, 3])

        # Orientation distance (geodesic angle on SO(3))
        R_delta = current_4x4[:3, :3].T @ target_4x4[:3, :3]
        cos_angle = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
        rot_angle = float(np.arccos(cos_angle))

        # Duration from position (1.5x safety factor for quintic peak velocity)
        pos_duration = pos_dist / (speed * 0.5) * 1.5

        # Duration from orientation (conservative angular speed)
        max_angular_vel = speed * 1.5  # rad/s
        rot_duration = rot_angle / max_angular_vel if max_angular_vel > 1e-6 else 0.0

        # Take the binding constraint, with raised minimum floor
        duration = float(max(1.0, pos_duration, rot_duration))

        self._control.start_cartesian_motion(self._robot, target_4x4, duration)
        if blocking:
            self._control.wait()

    def home(self, speed: float = 0.3, blocking: bool = False) -> None:
        """Move robot to home position [0,-pi/4,0,-3*pi/4,0,pi/2,pi/4].

        Args:
            speed: Speed fraction 0-1 (default 0.3).
            blocking: If True, block until motion completes.
        """
        gen = JointMotionGenerator(HOME_Q, speed_factor=speed)
        self._control.start_joint_motion(self._robot, gen)
        if blocking:
            self._control.wait()

    def set_velocity(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        wx: float = 0.0,
        wy: float = 0.0,
        wz: float = 0.0,
        smoother_params: dict | None = None,
    ) -> None:
        """Stream Cartesian velocity command (joystick-style control).

        Starts velocity control mode if not already running.

        Args:
            vx, vy, vz: Linear velocity in m/s (robot base frame).
            wx, wy, wz: Angular velocity in rad/s (robot base frame).
            smoother_params: Optional dict to configure the 1 kHz jerk-limited
                smoother.  Keys: ``K`` (float, default 15.0), ``max_accel``
                (list of 6 floats), ``max_jerk`` (list of 6 floats),
                ``command_timeout`` (seconds before stale velocity input is
                treated as zero; default 0.5).
                Only applied when the velocity thread starts (first call or
                after recovery).
        """
        if smoother_params is not None:
            self._smoother_params = smoother_params
        if not self._control.is_running():
            # Check if a previous velocity thread died with an error
            err = self._control.get_error()
            if err is not None:
                self._control._error = None
                raise RuntimeError(f"Velocity control failed: {err}") from err
            self._control.start_velocity(
                self._robot, smoother_params=self._smoother_params
            )
        self._control.set_velocity([vx, vy, vz], [wx, wy, wz])

    def set_joint_targets(
        self,
        targets: np.ndarray,
        speed: float | None = None,
        speed_fraction: float | None = None,
        speed_rad_s: float | None = None,
        duration: float | None = None,
        blocking: bool = False,
    ) -> None:
        """Move joints to target angles using S-curve trajectory.

        Exactly one of speed/speed_fraction, speed_rad_s, or duration can be set.
        """
        targets = np.array(targets, dtype=float)
        validate_joint_limits(targets)

        if speed is not None and speed_fraction is None:
            speed_fraction = speed

        provided = sum(x is not None for x in [speed_fraction, speed_rad_s, duration])
        if provided == 0:
            speed_fraction = 0.3
        elif provided > 1:
            raise ValueError(
                "Provide exactly one of: speed/speed_fraction, speed_rad_s, duration"
            )

        if speed_fraction is not None:
            gen = JointMotionGenerator.from_fraction(targets, speed_fraction)
        elif speed_rad_s is not None:
            gen = JointMotionGenerator.from_rad_s(targets, speed_rad_s)
        else:
            assert duration is not None
            current_q = self.get_joints()
            gen = JointMotionGenerator.from_duration(targets, current_q, duration)

        self._control.start_joint_motion(self._robot, gen)
        if blocking:
            self._control.wait()

    def start_streaming(
        self, speed: float = 0.3, drop_last: int = 0
    ) -> StreamingMotionGenerator:
        """Start streaming joint position control mode.

        Returns a StreamingMotionGenerator whose add_waypoints() method
        accepts (N, 7) arrays of joint configurations.  The robot holds its
        current position until waypoints are provided.  Call stop() to end.

        Args:
            speed: Speed fraction 0-1 (default 0.3).
            drop_last: When >0, each call to add_waypoints() first removes
                up to this many waypoints from the tail of the queue,
                enabling MPC-style rolling-horizon replanning.
        """
        gen = StreamingMotionGenerator(speed_factor=speed, drop_last=drop_last)
        self._streaming_gen = gen
        self._control.start_streaming_motion(self._robot, gen)
        return gen

    def send_joint_waypoints(self, waypoints: np.ndarray) -> None:
        """Send joint waypoints to an active streaming controller.

        Args:
            waypoints: Shape (7,) or (N, 7) joint configurations in radians.

        Raises:
            RuntimeError: If streaming mode is not active.
        """
        self.check_motion()
        if self._streaming_gen is None or not self._control.is_running():
            raise RuntimeError(
                "Streaming mode not active. Call start_streaming() first."
            )
        self._streaming_gen.add_waypoints(waypoints)

    def check_motion(self) -> None:
        """Raise if the background control thread has failed."""
        err = self._control.get_error()
        if err is not None:
            self._control._error = None
            raise RuntimeError(f"Motion failed: {err}") from err

    def is_moving(self) -> bool:
        """Return True if a motion is currently active."""
        return self._control.is_running()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until current motion completes.

        Args:
            timeout: Maximum seconds to wait (None waits forever).

        Returns:
            True if motion completed, False if timed out.
        """
        return self._control.wait(timeout)

    def stop(self) -> None:
        """Stop all motion immediately."""
        self._control.stop()
        self._streaming_gen = None

    def recover(self) -> None:
        """Recover from reflex errors and collision states."""
        self._robot.automatic_error_recovery()

    def set_collision_thresholds(
        self,
        torque_lower: list[float] | None = None,
        torque_upper: list[float] | None = None,
        force_lower: list[float] | None = None,
        force_upper: list[float] | None = None,
    ) -> None:
        """Set collision detection thresholds.

        Higher values make the robot less sensitive to contact forces.
        Useful for teleoperation where dynamic forces from fast direction
        changes can falsely trigger reflexes.

        Called with no arguments, restores the constructor defaults.

        Args:
            torque_lower: 7-element joint torque thresholds (Nm). Default [20]*7.
            torque_upper: 7-element joint torque thresholds (Nm). Default [40]*7.
            force_lower: 6-element Cartesian force thresholds (N/Nm). Default [10]*6.
            force_upper: 6-element Cartesian force thresholds (N/Nm). Default [20]*6.
        """
        self._robot.set_collision_behavior(
            torque_lower or [20.0] * 7,
            torque_upper or [40.0] * 7,
            force_lower or [10.0] * 6,
            force_upper or [20.0] * 6,
        )

    def move_gripper(self, width: float, speed: float = 0.1) -> bool:
        """Move gripper fingers to a target width. Blocking."""
        return self._gripper.move(width, speed)

    def open_gripper(self, speed: float = 0.1) -> bool:
        """Open gripper fully. Blocking."""
        return self.move_gripper(0.08, speed)

    def close_gripper(self, speed: float = 0.1) -> bool:
        """Close gripper fully. Blocking."""
        return self.move_gripper(0.0, speed)

    def grasp(
        self,
        width: float,
        speed: float = 0.1,
        force: float = 20.0,
        epsilon_inner: float = 0.005,
        epsilon_outer: float = 0.005,
    ) -> bool:
        """Grasp object with epsilon tolerance. Blocking.

        Args:
            width: Target grasp width in meters.
            speed: Closing speed in m/s.
            force: Grasping force in N.
            epsilon_inner: Tolerance below target width in meters.
            epsilon_outer: Tolerance above target width in meters.

        Returns:
            True if grasp succeeded.
        """
        return self._gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)

    def homing_gripper(self) -> None:
        """Run gripper homing sequence. Blocking."""
        self._gripper.homing()
