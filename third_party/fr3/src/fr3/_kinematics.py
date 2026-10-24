"""Euler angle ↔ rotation matrix conversion utilities."""

import numpy as np
from scipy.spatial.transform import Rotation
from fr3._types import Pose


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX intrinsic Euler angles to 3×3 rotation matrix.

    User provides (roll, pitch, yaw). Internally converts to scipy ZYX order
    [yaw, pitch, roll] (reversed!), so R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Args:
        roll:  Rotation around the new X axis, applied last (radians).
        pitch: Rotation around the new Y axis (radians).
        yaw:   Rotation around the Z axis, applied first (radians).

    Returns:
        3×3 rotation matrix.
    """
    return Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()


def matrix_to_euler(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert 3×3 rotation matrix to ZYX intrinsic Euler angles.

    Returns (roll, pitch, yaw) — note scipy's as_euler('ZYX') returns
    [yaw, pitch, roll] which is reversed here.

    Args:
        rotation_matrix: 3×3 orthogonal rotation matrix.

    Returns:
        (roll, pitch, yaw) in radians.
    """
    zyx = Rotation.from_matrix(rotation_matrix).as_euler("ZYX")
    yaw, pitch, roll = zyx[0], zyx[1], zyx[2]
    return roll, pitch, yaw


def pose_to_o_t_ee(pose: Pose) -> list[float]:
    """Convert a Pose to pylibfranka's 16-element column-major O_T_EE format.

    pylibfranka represents poses as 4×4 homogeneous transform matrices stored
    as 16 doubles in column-major (Fortran) order. Position is at indices
    [12, 13, 14] (confirmed by pylibfranka examples).

    Args:
        pose: End-effector pose in base frame.

    Returns:
        List of 16 floats in column-major order.
    """
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = euler_to_matrix(pose.roll, pose.pitch, pose.yaw)
    mat[:3, 3] = [pose.x, pose.y, pose.z]
    return list(mat.flatten(order="F"))


def o_t_ee_to_pose(o_t_ee: list[float]) -> Pose:
    """Convert pylibfranka's 16-element column-major O_T_EE to a Pose.

    Args:
        o_t_ee: 16-element list from pylibfranka RobotState.O_T_EE.

    Returns:
        Pose with position and ZYX Euler angles in radians.
    """
    mat = np.array(o_t_ee, dtype=float).reshape((4, 4), order="F")
    x, y, z = mat[:3, 3]
    roll, pitch, yaw = matrix_to_euler(mat[:3, :3])
    return Pose(
        x=float(x),
        y=float(y),
        z=float(z),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
    )
