"""Gripper control wrapper for pylibfranka.Gripper."""

from pylibfranka import Gripper


class GripperControl:
    """High-level wrapper around pylibfranka.Gripper.

    The gripper uses a separate TCP connection (port 1338), independent from
    the robot arm control (port 1337). Gripper methods CAN be called while
    the arm is moving. All gripper methods block until the command finishes.

    Example:
        gripper = GripperControl("<ROBOT_IP>")
        gripper.homing()
        gripper.open()
        gripper.grasp(width=0.04, epsilon_inner=0.04, epsilon_outer=0.04)
    """

    def __init__(self, ip: str):
        """Connect to gripper at given IP address."""
        self._gripper = Gripper(ip)

    def homing(self) -> None:
        """Run gripper homing sequence to calibrate max width. Blocking."""
        self._gripper.homing()

    def move(self, width: float, speed: float = 0.1) -> bool:
        """Move gripper fingers to specific width in meters. Blocking.

        Args:
            width: Target width in meters (0.0 = fully closed, 0.08 = fully open)
            speed: Movement speed in m/s (default 0.1)

        Returns:
            True if move succeeded
        """
        return self._gripper.move(width, speed)

    def open(self, speed: float = 0.1) -> bool:
        """Open gripper fully (width=0.08m). Blocking.

        Args:
            speed: Movement speed in m/s (default 0.1)
        """
        return self.move(0.08, speed)

    def close(self, speed: float = 0.1) -> bool:
        """Close gripper fully (width=0.0m). Blocking.

        Args:
            speed: Movement speed in m/s (default 0.1)
        """
        return self.move(0.0, speed)

    def grasp(
        self,
        width: float,
        speed: float = 0.1,
        force: float = 20.0,
        epsilon_inner: float = 0.005,
        epsilon_outer: float = 0.005,
    ) -> bool:
        """Grasp an object with epsilon tolerance. Blocking.

        Args:
            width: Target grasp width in meters
            speed: Closing speed in m/s (default 0.1)
            force: Grasping force in N (default 20.0)
            epsilon_inner: How much LESS than width fingers can be and still
                count as grasped (default 0.005m = 5mm)
            epsilon_outer: How much MORE than width fingers can be and still
                count as grasped (default 0.005m = 5mm)

        Returns:
            True if grasp succeeded (fingers within epsilon of target width)

        Example:
            # Grasp unknown object — accept any position as success
            robot.grasp(width=0.04, epsilon_inner=0.04, epsilon_outer=0.04)
        """
        return self._gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)

    def read_state(self) -> dict:
        """Read current gripper state.

        Returns:
            dict with keys: 'width' (float, meters), 'max_width' (float),
            'is_grasped' (bool)
        """
        state = self._gripper.read_once()
        return {
            "width": state.width,
            "max_width": getattr(state, "max_width", 0.08),
            "is_grasped": state.is_grasped,
        }

    def stop(self) -> None:
        """Stop gripper motion immediately."""
        self._gripper.stop()
