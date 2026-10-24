"""fr3 - High-level Python control library for Franka Research 3.

Usage:
    import fr3
    fr3.ROBOT_IP = "<ROBOT_IP>"  # Set once at top of script
    robot = fr3.FR3()            # Uses ROBOT_IP
    # or
    robot = fr3.FR3("<ROBOT_IP>")  # Explicit IP
"""

__version__ = "0.1.0"

ROBOT_IP: str | None = None

from fr3._gamepad import Gamepad, deadzone
from fr3._types import HOME_Q, JointState, Pose
from fr3.robot import FR3

__all__ = [
    "FR3",
    "Gamepad",
    "Pose",
    "JointState",
    "HOME_Q",
    "ROBOT_IP",
    "deadzone",
    "__version__",
]
