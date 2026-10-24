"""Core data types: Pose, JointState, constants."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Pose:
    """End-effector pose in robot base frame.

    Attributes:
        x, y, z: Position in metres.
        roll, pitch, yaw: ZYX intrinsic Euler angles in radians (yaw around Z,
                          then pitch around new Y, then roll around new X).
    """

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass
class JointState:
    """Current joint configuration.

    Attributes:
        positions: 7-element array of joint angles in radians (joints 1-7).
    """

    positions: np.ndarray


# Home position in radians (joints 1-7).
# Classic Franka "ready" configuration (flange pointing straight down).
HOME_Q = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])

NUM_JOINTS: int = 7

# Joint limits from fr3.xml range attributes (min, max) in radians
JOINT_LIMITS = np.array(
    [
        [-2.7437, 2.7437],  # joint 1
        [-1.7837, 1.7837],  # joint 2
        [-2.9007, 2.9007],  # joint 3
        [-3.0421, -0.1518],  # joint 4
        [-2.8065, 2.8065],  # joint 5
        [0.5445, 4.5169],  # joint 6
        [-3.0159, 3.0159],  # joint 7
    ]
)


def validate_joint_limits(q: np.ndarray) -> None:
    """Raise ValueError if any joint angle is outside JOINT_LIMITS."""
    q = np.atleast_2d(q)
    below = q < JOINT_LIMITS[:, 0]
    above = q > JOINT_LIMITS[:, 1]
    violations = np.argwhere(below | above)
    if violations.size:
        r, j = violations[0]
        raise ValueError(
            f"Joint {j + 1} target {q[r, j]:.4f} rad outside limits "
            f"[{JOINT_LIMITS[j, 0]:.4f}, {JOINT_LIMITS[j, 1]:.4f}]"
        )
