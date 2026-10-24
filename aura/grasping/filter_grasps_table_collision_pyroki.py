#!/usr/bin/env python3
r"""
Offline AnyGrasp table-collision + PyRoki IK/FK filter for Franka FR3.

This script does NOT connect to or command the robot. For each grasp it:
  1. converts the AnyGrasp pose to the world frame;
  2. rejects a grasp whose final TCP target (grasp center plus --final-z-offset
     along FR3 base z, which points up; default 0) is at/on/below world table z
     (default: require z < 0);
  3. applies the execution rotation convention (default: local +90 deg about Y);
  4. checks the calibrated fingertip points against the table and the tool
     points against the camera keep-outs at the final target;
  5. converts the TCP target to the Franka base frame;
  6. samples the exact direct Cartesian current->hover path used by execution;
  7. solves continuous PyRoki IK waypoints along that path using the previous
     solution as the next seed;
  8. verifies every waypoint with FK, joint margins, step continuity, and the
     calibrated fingertip/table and camera keep-out checks, then the
     --max-home-hover-* endpoint-delta, total-motion and rotation thresholds;
  9. solves and verifies HOVER/FINAL/LIFT endpoints and their short local paths.

The generated current-to-hover q_path is validation-only: it is never sent to
the robot. The executor keeps using CartesianPose control. This is not a full
robot-link/self/environment collision checker; table safety uses the two
calibrated fingertip points, while camera keep-outs use the TCP, fingertip
points, and TCP-to-tip midpoints. Use the generated report before executing.

Typical use (from the repository root):
  set -a; source config/aura.env; set +a
  conda activate "$AURA_ENV"
  python aura/grasping/filter_grasps_table_collision_pyroki.py \
      --grasps "$AURA_DATA_ROOT"/franka_lift/<case>/anygrasp_results/grasps.json \
      --calib-results "$AURA_CALIB_ROOT"/robot_calibration_results \
      --state-file assets/fr3_home_state.npz \
      --camera-extrinsic-dir "$AURA_CALIB_ROOT"/extrinsic_results \
      --out "$AURA_DATA_ROOT"/debug/<case>/grasps_table_ik_safe.json \
      --report "$AURA_DATA_ROOT"/debug/<case>/grasps_table_ik_report.json

--calib-results must hold both robot_calib_result.npz and
fingertip_calibration_results/franka_fingertip_table_calibration_simple.json
(see calibration/README.md). Without --camera-extrinsic-dir the camera
keep-out check is disabled. scripts/anygrasp_poses_one_shot.sh runs this script
as stage 04 with every threshold spelled out (the six options it omits keep this
script's defaults), including four robustness thresholds that are looser than
this script's defaults; copy its arguments to reproduce a pipeline run.

Pass --out and --report as above. Without them the outputs are written next to
the --grasps file (grasps_table_ik_safe.json, grasps_table_ik_report.json), so
a hand run on a case's anygrasp_results/grasps.json replaces the case's stage-04
safe list and report: anygrasp_poses_one_shot.sh --resume keeps them unless
--refilter is given, and the lift one-shot with --no-refilter executes them.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


FR3_SOFT_LIMITS_RAD: Mapping[str, Tuple[float, float]] = {
    "fr3_joint1": (-2.8007, +2.8007),
    "fr3_joint2": (-1.7361, +1.7361),
    "fr3_joint3": (-2.8007, +2.8007),
    "fr3_joint4": (-2.7500, -0.2200),
    "fr3_joint5": (-2.7763, +2.7763),
    "fr3_joint6": (+0.5398, +4.5216),
    "fr3_joint7": (-2.9508, +2.9508),
}


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def make_T(R: np.ndarray, t: Iterable[float]) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"Rotation must be 3x3, got {R.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def validate_T(T: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} contains non-finite values")
    R = T[:3, :3]
    det = float(np.linalg.det(R))
    ortho_error = float(np.linalg.norm(R.T @ R - np.eye(3)))
    if abs(det - 1.0) > 1e-2 or ortho_error > 1e-2:
        raise ValueError(
            f"{name} is not a valid rigid transform: det={det:.6f}, "
            f"orthonormal_error={ortho_error:.3e}"
        )
    return T


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized [w, x, y, z] quaternion."""
    R = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Could not convert rotation matrix to quaternion")
    q /= norm
    if q[0] < 0.0:
        q = -q
    return q.astype(np.float64)


def pose7_wxyz_xyz_to_T(pose: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(7)
    w, x, y, z = pose[:4]
    q_norm = math.sqrt(w * w + x * x + y * y + z * z)
    if q_norm < 1e-12:
        raise ValueError("FK returned a zero quaternion")
    w, x, y, z = np.array([w, x, y, z]) / q_norm
    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return make_T(R, pose[4:7])


def rotation_error_deg(R_actual: np.ndarray, R_target: np.ndarray) -> float:
    relative = R_actual.T @ R_target
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def quat_wxyz_to_R(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    w, x, y, z = np.asarray([w, x, y, z], dtype=np.float64) / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - alpha) * theta) / sin_theta * q0
        + math.sin(alpha * theta) / sin_theta * q1
    )


def interpolate_cartesian_segment(
    T_start: np.ndarray,
    T_end: np.ndarray,
    linear_step_m: float,
    angular_step_deg: float,
) -> List[np.ndarray]:
    """Sample a shortest-orientation Cartesian segment including both ends."""
    if linear_step_m <= 0.0 or angular_step_deg <= 0.0:
        raise ValueError("Cartesian path step sizes must be positive")
    T_start = validate_T(T_start, "Cartesian segment start")
    T_end = validate_T(T_end, "Cartesian segment end")
    distance = float(np.linalg.norm(T_end[:3, 3] - T_start[:3, 3]))
    angle_deg = rotation_error_deg(T_start[:3, :3], T_end[:3, :3])
    samples = max(
        2,
        int(math.ceil(distance / linear_step_m)) + 1,
        int(math.ceil(angle_deg / angular_step_deg)) + 1,
    )
    q0 = matrix_to_quat_wxyz(T_start[:3, :3])
    q1 = matrix_to_quat_wxyz(T_end[:3, :3])
    output: List[np.ndarray] = []
    for alpha in np.linspace(0.0, 1.0, samples, dtype=np.float64):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = (1.0 - alpha) * T_start[:3, 3] + alpha * T_end[:3, 3]
        T[:3, :3] = quat_wxyz_to_R(slerp_wxyz(q0, q1, float(alpha)))
        output.append(T)
    return output


def interpolate_cartesian_segment_samples(
    T_start: np.ndarray,
    T_end: np.ndarray,
    samples: int,
) -> List[np.ndarray]:
    """Sample a shortest-orientation Cartesian segment at a fixed count."""
    if samples < 2:
        raise ValueError("Cartesian path samples must be at least 2")
    T_start = validate_T(T_start, "Cartesian segment start")
    T_end = validate_T(T_end, "Cartesian segment end")
    q0 = matrix_to_quat_wxyz(T_start[:3, :3])
    q1 = matrix_to_quat_wxyz(T_end[:3, :3])
    output: List[np.ndarray] = []
    for alpha in np.linspace(0.0, 1.0, samples, dtype=np.float64):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = (1.0 - alpha) * T_start[:3, 3] + alpha * T_end[:3, 3]
        T[:3, :3] = quat_wxyz_to_R(slerp_wxyz(q0, q1, float(alpha)))
        output.append(T)
    output[0] = T_start.copy()
    output[-1] = T_end.copy()
    return output


def grasp_T(grasp: Mapping[str, Any]) -> np.ndarray:
    if "pose_matrix_4x4" in grasp:
        return validate_T(np.asarray(grasp["pose_matrix_4x4"], dtype=np.float64), "pose_matrix_4x4")
    if "rotation_matrix" in grasp and "translation" in grasp:
        return validate_T(make_T(grasp["rotation_matrix"], grasp["translation"]), "grasp pose")
    raise KeyError("Each grasp needs pose_matrix_4x4 or rotation_matrix + translation")


def apply_orientation_offset(T_input_grasp: np.ndarray, R_offset: np.ndarray, mode: str) -> np.ndarray:
    T = T_input_grasp.copy()
    if mode == "post":
        T[:3, :3] = T_input_grasp[:3, :3] @ R_offset
    elif mode == "pre":
        T[:3, :3] = R_offset @ T_input_grasp[:3, :3]
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported offset mode: {mode}")
    return T


def resolve_calib_root(calib_results: Path) -> Tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Accept a results directory, robot_calib_result.npz, or calibration ZIP."""
    calib_results = as_path(calib_results)
    temporary: tempfile.TemporaryDirectory[str] | None = None

    if calib_results.is_file() and calib_results.suffix.lower() == ".zip":
        temporary = tempfile.TemporaryDirectory(prefix="aura_calibration_")
        with zipfile.ZipFile(calib_results, "r") as archive:
            archive.extractall(temporary.name)
        search_root = Path(temporary.name)
    elif calib_results.is_file() and calib_results.suffix.lower() == ".npz":
        search_root = calib_results.parent
    elif calib_results.is_dir():
        search_root = calib_results
    else:
        raise FileNotFoundError(f"Calibration path does not exist or is unsupported: {calib_results}")

    candidates = [
        search_root,
        search_root / "results",
        search_root / "robot_calibration_results",
        search_root / "results" / "robot_calibration_results",
    ]
    candidates.extend(path.parent for path in search_root.rglob("robot_calib_result.npz"))
    candidates.extend(path.parent.parent for path in search_root.rglob("*fingertip*calibration*.json"))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        has_robot = (candidate / "robot_calib_result.npz").exists()
        has_tips = bool(list(candidate.rglob("*fingertip*calibration*.json")))
        if has_robot and has_tips:
            return candidate, temporary

    if temporary is not None:
        temporary.cleanup()
    raise FileNotFoundError(
        f"Could not find both robot_calib_result.npz and a fingertip calibration JSON under {calib_results}"
    )


def find_fingertip_json(calib_root: Path) -> Path:
    preferred = calib_root / "fingertip_calibration_results" / "franka_fingertip_table_calibration_simple.json"
    if preferred.exists():
        return preferred
    matches = sorted(calib_root.rglob("*fingertip*calibration*.json"))
    if not matches:
        raise FileNotFoundError(f"No fingertip calibration JSON found under {calib_root}")
    return matches[0]


def load_calibration(calib_root: Path, pose_frame: str) -> Dict[str, Any]:
    npz_path = calib_root / "robot_calib_result.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    data = np.load(npz_path)
    if "T_world_from_base" not in data.files:
        raise KeyError(f"{npz_path} needs T_world_from_base")
    T_world_base = validate_T(np.asarray(data["T_world_from_base"], dtype=np.float64), "T_world_from_base")

    if pose_frame == "world":
        T_world_input = np.eye(4, dtype=np.float64)
    elif pose_frame == "base":
        T_world_input = T_world_base
    elif pose_frame == "camera":
        if "T_world_from_camera" not in data.files:
            raise KeyError(f"{npz_path} needs T_world_from_camera for --pose-frame camera")
        T_world_input = validate_T(
            np.asarray(data["T_world_from_camera"], dtype=np.float64), "T_world_from_camera"
        )
    else:
        raise ValueError(f"Unsupported pose frame: {pose_frame}")

    tip_path = find_fingertip_json(calib_root)
    tip_data = load_json(tip_path)
    translations = tip_data.get("end_effector_fingertip_translation", {})
    required = ["right_outer_tip_in_ee_m", "left_outer_tip_in_ee_m"]
    missing = [key for key in required if key not in translations]
    if missing:
        raise KeyError(f"Missing fingertip fields in {tip_path}: {missing}")
    tips = {
        key.replace("_in_ee_m", ""): np.asarray(translations[key], dtype=np.float64).reshape(3)
        for key in required
    }
    return {
        "T_world_from_base": T_world_base,
        "T_base_from_world": np.linalg.inv(T_world_base),
        "T_world_from_input": T_world_input,
        "tips_in_ee": tips,
        "robot_calib_path": str(npz_path),
        "fingertip_calib_path": str(tip_path),
    }


def table_margin(z: float, table_z: float, z_positive_down: bool) -> float:
    return (table_z - z) if z_positive_down else (z - table_z)


def center_is_above_table(z: float, table_z: float, z_positive_down: bool) -> bool:
    return z < table_z if z_positive_down else z > table_z


def tip_table_check(
    T_world_ee: np.ndarray,
    tips_in_ee: Mapping[str, np.ndarray],
    table_z: float,
    clearance_m: float,
    z_positive_down: bool,
) -> Dict[str, Any]:
    positions: Dict[str, List[float]] = {}
    margins: Dict[str, float] = {}
    for name, p_ee in tips_in_ee.items():
        p_world = (T_world_ee @ np.r_[p_ee, 1.0])[:3]
        positions[name] = p_world.tolist()
        margins[name] = table_margin(float(p_world[2]), table_z, z_positive_down)
    worst_tip = min(margins, key=margins.get)
    worst_margin = float(margins[worst_tip])
    return {
        "safe": bool(worst_margin >= clearance_m),
        "worst_tip": worst_tip,
        "minimum_clearance_m": worst_margin,
        "required_clearance_m": float(clearance_m),
        "tip_positions_world_m": positions,
        "tip_clearances_m": margins,
    }


def load_camera_keepouts(extrinsic_dir: Path | None) -> List[Dict[str, Any]]:
    if extrinsic_dir is None:
        return []
    root = as_path(extrinsic_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Camera extrinsic directory not found: {root}")
    cameras: List[Dict[str, Any]] = []
    loaded_names: set[str] = set()
    for path in sorted(root.glob("*.npz")):
        if "oak_" not in path.name:
            continue
        data = np.load(path)
        if "T_world_from_cam" in data.files:
            T_world_camera = validate_T(data["T_world_from_cam"], f"{path}:T_world_from_cam")
        elif "phys_twin_c2w" in data.files:
            T_world_camera = validate_T(data["phys_twin_c2w"], f"{path}:phys_twin_c2w")
        elif "T_world_from_camera" in data.files:
            T_world_camera = validate_T(data["T_world_from_camera"], f"{path}:T_world_from_camera")
        elif "T_camera_from_world" in data.files:
            T_world_camera = np.linalg.inv(
                validate_T(data["T_camera_from_world"], f"{path}:T_camera_from_world")
            )
        else:
            continue
        cameras.append(
            {
                "name": path.stem,
                "source_path": str(path),
                "center_world_m": T_world_camera[:3, 3].tolist(),
                "T_world_from_camera": T_world_camera.tolist(),
            }
        )
        loaded_names.add(path.stem)

    # Some calibration runs preserve the same 4x4 transforms as JSON rather
    # than NPZ. Prefer NPZ when both exist, but accept the common JSON keys so
    # camera filtering does not silently depend on one serialization format.
    for path in sorted(root.glob("*.json")):
        if "oak_" not in path.name or path.stem in loaded_names:
            continue
        value = load_json(path)
        if not isinstance(value, Mapping):
            continue
        transform_value = None
        inverse = False
        for key in (
            "T_world_from_cam",
            "phys_twin_c2w",
            "T_world_from_camera",
            "camera_to_world",
        ):
            if key in value:
                transform_value = value[key]
                break
        if transform_value is None and "T_camera_from_world" in value:
            transform_value = value["T_camera_from_world"]
            inverse = True
        if transform_value is None:
            continue
        T_world_camera = validate_T(
            np.asarray(transform_value, dtype=np.float64),
            f"{path}:camera extrinsic",
        )
        if inverse:
            T_world_camera = np.linalg.inv(T_world_camera)
        cameras.append(
            {
                "name": path.stem,
                "source_path": str(path),
                "center_world_m": T_world_camera[:3, 3].tolist(),
                "T_world_from_camera": T_world_camera.tolist(),
            }
        )
        loaded_names.add(path.stem)
    if not cameras:
        raise FileNotFoundError(
            f"No recognized OAK camera-to-world transform in NPZ/JSON files under {root}"
        )
    return cameras


def tool_camera_keepout_check(
    T_world_ee: np.ndarray,
    tips_in_ee: Mapping[str, np.ndarray],
    cameras: Sequence[Mapping[str, Any]],
    keepout_radius_m: float,
) -> Dict[str, Any]:
    if not cameras or keepout_radius_m <= 0.0:
        return {
            "enabled": False,
            "safe": True,
            "keepout_radius_m": float(keepout_radius_m),
            "minimum_distance_m": None,
            "closest_camera": None,
            "closest_tool_point": None,
        }
    tool_points_ee: Dict[str, np.ndarray] = {"tcp": np.zeros(3, dtype=np.float64)}
    for name, point in tips_in_ee.items():
        point = np.asarray(point, dtype=np.float64).reshape(3)
        tool_points_ee[name] = point
        tool_points_ee[f"mid_{name}"] = 0.5 * point

    minimum = math.inf
    closest_camera = None
    closest_tool_point = None
    distances: Dict[str, Dict[str, float]] = {}
    for camera in cameras:
        camera_name = str(camera["name"])
        center = np.asarray(camera["center_world_m"], dtype=np.float64).reshape(3)
        distances[camera_name] = {}
        for point_name, point_ee in tool_points_ee.items():
            point_world = (T_world_ee @ np.r_[point_ee, 1.0])[:3]
            distance = float(np.linalg.norm(point_world - center))
            distances[camera_name][point_name] = distance
            if distance < minimum:
                minimum = distance
                closest_camera = camera_name
                closest_tool_point = point_name
    return {
        "enabled": True,
        "safe": bool(minimum >= float(keepout_radius_m)),
        "keepout_radius_m": float(keepout_radius_m),
        "minimum_distance_m": float(minimum),
        "closest_camera": closest_camera,
        "closest_tool_point": closest_tool_point,
        "distances_m": distances,
        "scope": "TCP, calibrated fingertip points, and TCP-to-tip midpoints; not full robot links",
    }


def load_current_arm_q(state_file: Path) -> Tuple[np.ndarray, str]:
    state_file = as_path(state_file)
    if not state_file.exists():
        raise FileNotFoundError(f"State file not found: {state_file}")
    data = np.load(state_file)
    candidates = ("q", "joint_positions", "joints", "q_measured", "measured_q")
    for key in candidates:
        if key in data.files:
            q = np.asarray(data[key], dtype=np.float64).reshape(-1)
            if q.size not in (7, 8):
                raise ValueError(f"State field {key!r} must contain 7 or 8 joints; got {q.size}")
            if not np.all(np.isfinite(q)):
                raise ValueError(f"State field {key!r} contains non-finite values")
            return q, key
    raise KeyError(f"No joint state found in {state_file}; tried keys {candidates}, available={data.files}")


@dataclass
class PyRokiContext:
    robot: Any
    jnp: Any
    solve_jax: Any
    target_link_index: int
    target_link: str
    actuated_names: Tuple[str, ...]
    current_q: np.ndarray
    lower_limits: np.ndarray
    upper_limits: np.ndarray


def build_pyroki_context(
    description: str,
    target_link: str,
    current_state_q: np.ndarray,
    open_finger_q: float,
    pos_weight: float,
    ori_weight: float,
    rest_weight: float,
) -> PyRokiContext:
    """Load FR3 and build a warm-started PyRoki IK solver."""
    try:
        import jax.numpy as jnp
        import jax_dataclasses as jdc
        import jaxlie
        import jaxls
        import pyroki as pk
        from robot_descriptions.loaders.yourdfpy import load_robot_description
    except ImportError as exc:
        raise RuntimeError(
            "Missing PyRoki dependencies (jax, jax_dataclasses, jaxlie, jaxls, pyroki, robot_descriptions). "
            "Activate the AuRA conda environment ($AURA_ENV in config/aura.env) before running this script."
        ) from exc

    urdf = load_robot_description(description)
    robot = pk.Robot.from_urdf(urdf)
    actuated_names = tuple(str(name) for name in robot.joints.actuated_names)
    expected_arm_names = tuple(f"fr3_joint{i}" for i in range(1, 8))
    missing = [name for name in expected_arm_names if name not in actuated_names]
    if missing:
        raise RuntimeError(f"Loaded robot does not contain expected FR3 arm joints: {missing}")
    if target_link not in robot.links.names:
        raise KeyError(f"Target link {target_link!r} not found. Available links: {robot.links.names}")
    target_link_index = int(robot.links.names.index(target_link))

    # Older PyRoki releases do not expose Robot.default_joint_cfg.  A
    # zero-initialized actuated-joint vector is sufficient here because every
    # FR3 arm joint is populated from the measured state below, and the gripper
    # joint is set explicitly.
    q_full = np.zeros(len(actuated_names), dtype=np.float64)
    if current_state_q.size == len(actuated_names):
        q_full[:] = current_state_q
    else:
        for arm_index, joint_name in enumerate(expected_arm_names):
            q_full[actuated_names.index(joint_name)] = current_state_q[arm_index]
        if "fr3_finger_joint1" in actuated_names:
            q_full[actuated_names.index("fr3_finger_joint1")] = float(open_finger_q)

    lower_limits = np.asarray(robot.joints.lower_limits, dtype=np.float64).reshape(-1)
    upper_limits = np.asarray(robot.joints.upper_limits, dtype=np.float64).reshape(-1)
    if lower_limits.size != q_full.size or upper_limits.size != q_full.size:
        raise RuntimeError("PyRoki joint-limit arrays do not match the actuated joint count")

    # The IK solve follows PyRoki's examples/pyroki_snippets/_solve_ik.py at the
    # pinned commit 388e43e (https://github.com/chungmin99/pyroki, MIT License,
    # Copyright (c) 2025 Chung Min Kim). AuRA adds the rest cost and the warm start.
    @jdc.jit
    def solve_jax(target_wxyz, target_position, previous_cfg):
        joint_var = robot.joint_var_cls(0)
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_wxyz), target_position
        )
        costs = [
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                target_pose,
                # Older PyRoki versions inspect ``target_link_index.shape``;
                # use a scalar JAX array rather than a plain Python int.
                jnp.asarray(target_link_index, dtype=jnp.int32),
                pos_weight=pos_weight,
                ori_weight=ori_weight,
            ),
            pk.costs.rest_cost(joint_var, previous_cfg, weight=rest_weight),
            pk.costs.limit_constraint(robot, joint_var),
        ]
        problem = jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var]).analyze()
        initial_values = jaxls.VarValues.make([joint_var.with_value(previous_cfg)])
        solution = problem.solve(
            initial_vals=initial_values,
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
        return solution[joint_var]

    # Make sure JAX sees one predictable floating-point dtype.
    current_q = np.asarray(q_full, dtype=np.float64)
    return PyRokiContext(
        robot=robot,
        jnp=jnp,
        solve_jax=solve_jax,
        target_link_index=target_link_index,
        target_link=target_link,
        actuated_names=actuated_names,
        current_q=current_q,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )


def fk_target_T(context: PyRokiContext, q: np.ndarray) -> np.ndarray:
    poses = np.asarray(
        context.robot.forward_kinematics(context.jnp.asarray(q)), dtype=np.float64
    )
    return pose7_wxyz_xyz_to_T(poses[context.target_link_index])


def hard_limit_check(context: PyRokiContext, q: np.ndarray, tolerance: float = 1e-7) -> Dict[str, Any]:
    q = np.asarray(q, dtype=np.float64)
    margins = np.minimum(q - context.lower_limits, context.upper_limits - q)
    worst = int(np.argmin(margins))
    return {
        "safe": bool(np.all(margins >= -tolerance)),
        "minimum_margin_rad": float(margins[worst]),
        "worst_joint": context.actuated_names[worst],
    }


def soft_limit_check(
    context: PyRokiContext,
    q: np.ndarray,
    enabled: bool,
    required_margin_rad: float = 0.0,
) -> Dict[str, Any]:
    if not enabled:
        return {"safe": True, "enabled": False, "minimum_margin_rad": None, "worst_joint": None}
    margins: Dict[str, float] = {}
    for joint_name, (lower, upper) in FR3_SOFT_LIMITS_RAD.items():
        index = context.actuated_names.index(joint_name)
        margins[joint_name] = min(float(q[index]) - lower, upper - float(q[index]))
    worst_joint = min(margins, key=margins.get)
    return {
        "safe": bool(margins[worst_joint] >= float(required_margin_rad)),
        "enabled": True,
        "required_margin_rad": float(required_margin_rad),
        "minimum_margin_rad": float(margins[worst_joint]),
        "worst_joint": worst_joint,
        "margins_rad": margins,
    }


def solve_and_verify_endpoint(
    context: PyRokiContext,
    target_T_base: np.ndarray,
    seed_q: np.ndarray,
    max_position_error_m: float,
    max_orientation_error_deg: float,
    use_soft_limits: bool,
    required_soft_margin_rad: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    target_q = matrix_to_quat_wxyz(target_T_base[:3, :3])
    solution = context.solve_jax(
        context.jnp.asarray(target_q),
        context.jnp.asarray(target_T_base[:3, 3]),
        context.jnp.asarray(seed_q),
    )
    q = np.array(solution, dtype=np.float64, copy=True)
    if q.shape != seed_q.shape or not np.all(np.isfinite(q)):
        raise RuntimeError(f"IK returned invalid configuration with shape {q.shape}")

    actual_T = fk_target_T(context, q)
    position_error_m = float(np.linalg.norm(actual_T[:3, 3] - target_T_base[:3, 3]))
    orientation_error_deg = rotation_error_deg(actual_T[:3, :3], target_T_base[:3, :3])
    hard = hard_limit_check(context, q)
    soft = soft_limit_check(context, q, use_soft_limits, required_soft_margin_rad)
    pose_ok = position_error_m <= max_position_error_m and orientation_error_deg <= max_orientation_error_deg
    result = {
        "safe": bool(pose_ok and hard["safe"] and soft["safe"]),
        "q_all_actuated_rad": q.tolist(),
        "q_arm_rad": [float(q[context.actuated_names.index(f"fr3_joint{i}")]) for i in range(1, 8)],
        "target_T_base_from_tcp": target_T_base.tolist(),
        "fk_T_base_from_tcp": actual_T.tolist(),
        "position_error_m": position_error_m,
        "orientation_error_deg": orientation_error_deg,
        "pose_tolerance_ok": bool(pose_ok),
        "hard_joint_limits": hard,
        "soft_joint_limits": soft,
    }
    return q, result


def arm_q(context: PyRokiContext, q: np.ndarray) -> np.ndarray:
    return np.asarray(
        [q[context.actuated_names.index(f"fr3_joint{i}")] for i in range(1, 8)],
        dtype=np.float64,
    )


def evaluate_direct_cartesian_hover_robustness(
    context: PyRokiContext,
    hover_target_T_base: np.ndarray,
    calibration: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[List[np.ndarray] | None, Dict[str, Any]]:
    """Evaluate the same direct Cartesian Home->Hover path used by execution."""
    current_T = fk_target_T(context, context.current_q)
    direct_segment = interpolate_cartesian_segment(
        current_T,
        hover_target_T_base,
        args.hover_path_linear_step_m,
        args.hover_path_angular_step_deg,
    )
    targets: List[Tuple[str, np.ndarray]] = [
        ("start", direct_segment[0]),
        *[("direct_cartesian_to_hover", T) for T in direct_segment[1:]],
    ]

    q_path: List[np.ndarray] = [context.current_q.copy()]
    current_actual_T = fk_target_T(context, context.current_q)
    current_table = tip_table_check(
        calibration["T_world_from_base"] @ current_actual_T,
        calibration["tips_in_ee"],
        args.table_z,
        args.clearance_m,
        args.z_positive_down,
    )
    current_camera = tool_camera_keepout_check(
        calibration["T_world_from_base"] @ current_actual_T,
        calibration["tips_in_ee"],
        calibration.get("camera_keepouts", []),
        args.camera_keepout_radius_m,
    )
    start_hard = hard_limit_check(context, context.current_q)
    start_soft = soft_limit_check(
        context,
        context.current_q,
        args.use_soft_limits,
        args.robust_joint_margin_rad,
    )
    waypoint_reports: List[Dict[str, Any]] = [
        {
            "index": 0,
            "stage": "start",
            "safe": bool(
                start_hard["safe"] and start_soft["safe"]
                and current_table["safe"] and current_camera["safe"]
            ),
            "q_arm_rad": arm_q(context, context.current_q).tolist(),
            "target_T_base_from_tcp": current_T.tolist(),
            "fk_T_base_from_tcp": current_actual_T.tolist(),
            "position_error_m": 0.0,
            "orientation_error_deg": 0.0,
            "max_step_joint_delta_rad": 0.0,
            "hard_joint_limits": start_hard,
            "soft_joint_limits": start_soft,
            "fingertip_table_check": current_table,
            "camera_keepout_check": current_camera,
        }
    ]
    if not waypoint_reports[0]["safe"]:
        return None, {
            "safe": False,
            "path_type": "direct_cartesian_path_dense_ik_robustness_filter_only",
            "reason": "Current/start state failed joint-margin, table, or camera keep-out validation",
            "waypoints": waypoint_reports,
        }

    seed = context.current_q.copy()
    failure_reason: str | None = None
    for waypoint_index, (stage, target_T) in enumerate(targets[1:], start=1):
        q, endpoint = solve_and_verify_endpoint(
            context,
            target_T,
            seed,
            args.max_position_error_m,
            args.max_orientation_error_deg,
            args.use_soft_limits,
            args.robust_joint_margin_rad,
        )
        previous_arm = arm_q(context, seed)
        current_arm = arm_q(context, q)
        max_step = float(np.max(np.abs(current_arm - previous_arm)))
        actual_T = np.asarray(endpoint["fk_T_base_from_tcp"], dtype=np.float64)
        table = tip_table_check(
            calibration["T_world_from_base"] @ actual_T,
            calibration["tips_in_ee"],
            args.table_z,
            args.clearance_m,
            args.z_positive_down,
        )
        camera = tool_camera_keepout_check(
            calibration["T_world_from_base"] @ actual_T,
            calibration["tips_in_ee"],
            calibration.get("camera_keepouts", []),
            args.camera_keepout_radius_m,
        )
        step_ok = max_step <= float(args.max_waypoint_joint_delta_rad)
        safe = bool(endpoint["safe"] and table["safe"] and camera["safe"] and step_ok)
        waypoint = {
            "index": waypoint_index,
            "stage": stage,
            "safe": safe,
            "q_arm_rad": current_arm.tolist(),
            "target_T_base_from_tcp": target_T.tolist(),
            "fk_T_base_from_tcp": actual_T.tolist(),
            "position_error_m": endpoint["position_error_m"],
            "orientation_error_deg": endpoint["orientation_error_deg"],
            "max_step_joint_delta_rad": max_step,
            "max_step_joint_delta_ok": step_ok,
            "hard_joint_limits": endpoint["hard_joint_limits"],
            "soft_joint_limits": endpoint["soft_joint_limits"],
            "fingertip_table_check": table,
            "camera_keepout_check": camera,
        }
        waypoint_reports.append(waypoint)
        if not safe:
            problems = []
            if not endpoint["safe"]:
                problems.append("IK/FK or joint-margin check")
            if not table["safe"]:
                problems.append("fingertip/table clearance")
            if not camera["safe"]:
                problems.append("camera keep-out")
            if not step_ok:
                problems.append(
                    f"joint branch jump {max_step:.4f}rad > {float(args.max_waypoint_joint_delta_rad):.4f}rad"
                )
            failure_reason = f"Waypoint {waypoint_index} ({stage}) failed: " + ", ".join(problems)
            break
        q_path.append(q.copy())
        seed = q

    q_arm_path = np.stack([arm_q(context, q) for q in q_path], axis=0)
    step_deltas = np.abs(np.diff(q_arm_path, axis=0)) if len(q_arm_path) > 1 else np.zeros((0, 7))
    total_joint_motion = float(np.sum(step_deltas))
    max_step_delta = float(np.max(step_deltas)) if step_deltas.size else 0.0
    minimum_table_clearance = min(
        float(w["fingertip_table_check"]["minimum_clearance_m"])
        for w in waypoint_reports
    )
    camera_distances = [
        float(w["camera_keepout_check"]["minimum_distance_m"])
        for w in waypoint_reports
        if w["camera_keepout_check"]["minimum_distance_m"] is not None
    ]
    minimum_soft_margin = min(
        float(w["soft_joint_limits"]["minimum_margin_rad"])
        for w in waypoint_reports
        if w["soft_joint_limits"]["minimum_margin_rad"] is not None
    ) if args.use_soft_limits else None
    endpoint_max_delta = float(np.max(np.abs(q_arm_path[-1] - q_arm_path[0])))
    cartesian_rotation_deg = rotation_error_deg(current_T[:3, :3], hover_target_T_base[:3, :3])
    robustness_problems: List[str] = []
    if endpoint_max_delta > float(args.max_home_hover_joint_delta_rad):
        robustness_problems.append(
            f"endpoint joint delta {endpoint_max_delta:.4f}rad > "
            f"{float(args.max_home_hover_joint_delta_rad):.4f}rad"
        )
    if total_joint_motion > float(args.max_home_hover_total_joint_motion_rad):
        robustness_problems.append(
            f"total joint motion {total_joint_motion:.4f}rad > "
            f"{float(args.max_home_hover_total_joint_motion_rad):.4f}rad"
        )
    if cartesian_rotation_deg > float(args.max_home_hover_rotation_deg):
        robustness_problems.append(
            f"Cartesian rotation {cartesian_rotation_deg:.2f}deg > "
            f"{float(args.max_home_hover_rotation_deg):.2f}deg"
        )
    if failure_reason is None and robustness_problems:
        failure_reason = "Robustness thresholds failed: " + "; ".join(robustness_problems)

    report = {
        "safe": failure_reason is None and len(q_path) == len(targets),
        "path_type": "direct_cartesian_path_dense_ik_robustness_filter_only",
        "execution_mode": "CartesianPose; q waypoints are validation-only and are not commanded",
        "stages": ["current_to_hover_direct_cartesian"],
        "linear_step_m": float(args.hover_path_linear_step_m),
        "angular_step_deg": float(args.hover_path_angular_step_deg),
        "required_joint_margin_rad": float(args.robust_joint_margin_rad),
        "max_allowed_waypoint_joint_delta_rad": float(args.max_waypoint_joint_delta_rad),
        "max_allowed_endpoint_joint_delta_rad": float(args.max_home_hover_joint_delta_rad),
        "max_allowed_total_joint_motion_rad": float(args.max_home_hover_total_joint_motion_rad),
        "max_allowed_cartesian_rotation_deg": float(args.max_home_hover_rotation_deg),
        "total_joint_motion_l1_rad": total_joint_motion,
        "max_waypoint_joint_delta_rad": max_step_delta,
        "endpoint_max_joint_delta_rad": endpoint_max_delta,
        "cartesian_rotation_deg": cartesian_rotation_deg,
        "minimum_soft_joint_margin_rad": minimum_soft_margin,
        "minimum_tip_table_clearance_m": minimum_table_clearance,
        "minimum_camera_distance_m": min(camera_distances) if camera_distances else None,
        "reason": failure_reason,
        "waypoints": waypoint_reports,
    }
    return (q_path if report["safe"] else None), report


def evaluate_direct_cartesian_local_segment(
    context: PyRokiContext,
    segment_name: str,
    start_T_base: np.ndarray,
    end_T_base: np.ndarray,
    seed_q: np.ndarray,
    calibration: Mapping[str, Any],
    args: argparse.Namespace,
) -> Tuple[np.ndarray | None, Dict[str, Any] | None, Dict[str, Any]]:
    """Densely validate one executed Cartesian segment without commanding q."""
    target_path = interpolate_cartesian_segment_samples(
        start_T_base,
        end_T_base,
        args.path_samples,
    )
    q_previous = np.asarray(seed_q, dtype=np.float64).copy()
    waypoint_reports: List[Dict[str, Any]] = []
    failure_reason: str | None = None
    final_endpoint: Dict[str, Any] | None = None

    for index, target_T in enumerate(target_path):
        if index == 0:
            actual_T = fk_target_T(context, q_previous)
            position_error_m = float(np.linalg.norm(actual_T[:3, 3] - target_T[:3, 3]))
            orientation_error_deg = rotation_error_deg(actual_T[:3, :3], target_T[:3, :3])
            hard = hard_limit_check(context, q_previous)
            soft = soft_limit_check(
                context,
                q_previous,
                args.use_soft_limits,
                args.robust_joint_margin_rad,
            )
            endpoint: Dict[str, Any] = {
                "safe": bool(
                    position_error_m <= args.max_position_error_m
                    and orientation_error_deg <= args.max_orientation_error_deg
                    and hard["safe"]
                    and soft["safe"]
                ),
                "q_all_actuated_rad": q_previous.tolist(),
                "q_arm_rad": arm_q(context, q_previous).tolist(),
                "target_T_base_from_tcp": target_T.tolist(),
                "fk_T_base_from_tcp": actual_T.tolist(),
                "position_error_m": position_error_m,
                "orientation_error_deg": orientation_error_deg,
                "hard_joint_limits": hard,
                "soft_joint_limits": soft,
            }
            max_step = 0.0
        else:
            q, endpoint = solve_and_verify_endpoint(
                context,
                target_T,
                q_previous,
                args.max_position_error_m,
                args.max_orientation_error_deg,
                args.use_soft_limits,
                args.robust_joint_margin_rad,
            )
            max_step = float(np.max(np.abs(arm_q(context, q) - arm_q(context, q_previous))))
            q_previous = q
            actual_T = np.asarray(endpoint["fk_T_base_from_tcp"], dtype=np.float64)

        table = tip_table_check(
            calibration["T_world_from_base"] @ actual_T,
            calibration["tips_in_ee"],
            args.table_z,
            args.clearance_m,
            args.z_positive_down,
        )
        camera = tool_camera_keepout_check(
            calibration["T_world_from_base"] @ actual_T,
            calibration["tips_in_ee"],
            calibration.get("camera_keepouts", []),
            args.camera_keepout_radius_m,
        )
        step_ok = max_step <= float(args.max_waypoint_joint_delta_rad)
        safe = bool(endpoint["safe"] and table["safe"] and camera["safe"] and step_ok)
        waypoint_reports.append(
            {
                "index": index,
                "safe": safe,
                "q_arm_rad": endpoint["q_arm_rad"],
                "target_T_base_from_tcp": target_T.tolist(),
                "fk_T_base_from_tcp": actual_T.tolist(),
                "position_error_m": endpoint["position_error_m"],
                "orientation_error_deg": endpoint["orientation_error_deg"],
                "max_step_joint_delta_rad": max_step,
                "max_step_joint_delta_ok": step_ok,
                "hard_joint_limits": endpoint["hard_joint_limits"],
                "soft_joint_limits": endpoint["soft_joint_limits"],
                "fingertip_table_check": table,
                "camera_keepout_check": camera,
            }
        )
        final_endpoint = endpoint
        if not safe:
            problems: List[str] = []
            if not endpoint["safe"]:
                problems.append("IK/FK or joint-margin check")
            if not table["safe"]:
                problems.append("fingertip/table clearance")
            if not camera["safe"]:
                problems.append("camera keep-out")
            if not step_ok:
                problems.append("IK branch jump")
            failure_reason = f"{segment_name} sample {index} failed: " + ", ".join(problems)
            break

    camera_distances = [
        float(w["camera_keepout_check"]["minimum_distance_m"])
        for w in waypoint_reports
        if w["camera_keepout_check"]["minimum_distance_m"] is not None
    ]
    report = {
        "safe": failure_reason is None and len(waypoint_reports) == len(target_path),
        "name": segment_name,
        "path_type": "executed_direct_cartesian_path_dense_ik_filter_only",
        "execution_mode": "CartesianPose; q waypoints are validation-only and are not commanded",
        "samples": int(args.path_samples),
        "reason": failure_reason,
        "minimum_tip_table_clearance_m": min(
            float(w["fingertip_table_check"]["minimum_clearance_m"])
            for w in waypoint_reports
        ),
        "minimum_camera_distance_m": min(camera_distances) if camera_distances else None,
        "maximum_waypoint_joint_delta_rad": max(
            float(w["max_step_joint_delta_rad"]) for w in waypoint_reports
        ),
        "waypoints": waypoint_reports,
    }
    return (
        q_previous if report["safe"] else None,
        final_endpoint if report["safe"] else None,
        report,
    )


def process_grasp(
    index: int,
    grasp: Mapping[str, Any],
    context: PyRokiContext,
    calibration: Mapping[str, Any],
    R_offset: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    entry: Dict[str, Any] = {
        "index": index,
        "rank": grasp.get("rank", index),
        "score": grasp.get("score"),
        "accepted": False,
        "rejection_stage": None,
        "reason": None,
    }
    try:
        T_input_grasp = grasp_T(grasp)
        T_world_grasp = calibration["T_world_from_input"] @ T_input_grasp
        T_world_tcp_nominal = apply_orientation_offset(
            T_world_grasp, R_offset, args.offset_mode
        )
        T_base_nominal = calibration["T_base_from_world"] @ T_world_tcp_nominal
        targets: Dict[str, np.ndarray] = {}
        for stage, z_offset in (
            ("hover", args.hover_z_offset),
            ("final", args.final_z_offset),
            ("lift", args.lift_z_offset),
        ):
            target = T_base_nominal.copy()
            target[2, 3] += float(z_offset)
            targets[stage] = target

        # All target safety gates must evaluate the pose the executor will
        # actually command. In particular, --final-z-offset is expressed in
        # base z and therefore must be applied before world-table/camera tests.
        T_world_tcp_final = calibration["T_world_from_base"] @ targets["final"]
        center_z = float(T_world_tcp_final[2, 3])
        center_ok = center_is_above_table(center_z, args.table_z, args.z_positive_down)
        entry["coarse_grasp_center_check"] = {
            "enabled": bool(args.require_center_above_table),
            "safe": bool(center_ok or not args.require_center_above_table),
            "center_world_m": T_world_tcp_final[:3, 3].tolist(),
            "nominal_center_world_m": T_world_grasp[:3, 3].tolist(),
            "final_z_offset_base_m": float(args.final_z_offset),
            "table_z_m": float(args.table_z),
            "criterion": "z < table_z" if args.z_positive_down else "z > table_z",
        }
        if args.require_center_above_table and not center_ok:
            entry["rejection_stage"] = "coarse_world_z"
            entry["reason"] = "Grasp center is at or below the table plane"
            return None, entry

        target_tip_check = tip_table_check(
            T_world_tcp_final,
            calibration["tips_in_ee"],
            args.table_z,
            args.clearance_m,
            args.z_positive_down,
        )
        entry["target_fingertip_table_check"] = target_tip_check
        entry["nominal_target_T_world_from_tcp"] = T_world_tcp_nominal.tolist()
        entry["target_T_world_from_tcp"] = T_world_tcp_final.tolist()
        entry["final_z_offset_base_m"] = float(args.final_z_offset)
        if not target_tip_check["safe"]:
            entry["rejection_stage"] = "target_fingertip_table"
            entry["reason"] = "Calibrated fingertip point violates table clearance at the final target"
            return None, entry

        target_camera_check = tool_camera_keepout_check(
            T_world_tcp_final,
            calibration["tips_in_ee"],
            calibration.get("camera_keepouts", []),
            args.camera_keepout_radius_m,
        )
        entry["target_camera_keepout_check"] = target_camera_check
        if not target_camera_check["safe"]:
            entry["rejection_stage"] = "target_camera_keepout"
            entry["reason"] = (
                "TCP/gripper geometry enters a camera keep-out sphere at the final target"
            )
            return None, entry

        q_hover_path, hover_path_report = evaluate_direct_cartesian_hover_robustness(
            context,
            targets["hover"],
            calibration,
            args,
        )
        entry["home_to_hover_cartesian_robustness"] = hover_path_report
        if q_hover_path is None:
            entry["rejection_stage"] = "home_to_hover_cartesian_robustness"
            entry["reason"] = hover_path_report.get("reason") or "Direct Cartesian path is not robust"
            return None, entry

        q_by_stage: Dict[str, np.ndarray] = {"hover": q_hover_path[-1]}
        endpoint_reports: Dict[str, Any] = {}
        q_hover_verified, hover_endpoint = solve_and_verify_endpoint(
            context,
            targets["hover"],
            q_by_stage["hover"],
            args.max_position_error_m,
            args.max_orientation_error_deg,
            args.use_soft_limits,
            args.robust_joint_margin_rad,
        )
        q_by_stage["hover"] = q_hover_verified
        endpoint_reports["hover"] = hover_endpoint
        if not hover_endpoint["safe"]:
            entry["ik_fk_endpoints"] = endpoint_reports
            entry["rejection_stage"] = "ik_fk_hover"
            entry["reason"] = "HOVER endpoint verification failed after continuous path solve"
            return None, entry

        seed = q_by_stage["hover"]
        local_segment_reports: List[Dict[str, Any]] = []
        previous_stage = "hover"
        for stage in ("final", "lift"):
            q, endpoint, segment_report = evaluate_direct_cartesian_local_segment(
                context,
                f"{previous_stage}_to_{stage}",
                targets[previous_stage],
                targets[stage],
                seed,
                calibration,
                args,
            )
            local_segment_reports.append(segment_report)
            if q is None or endpoint is None:
                entry["ik_fk_endpoints"] = endpoint_reports
                entry["cartesian_local_path_robustness"] = {
                    "safe": False,
                    "segments": local_segment_reports,
                }
                entry["rejection_stage"] = f"cartesian_{previous_stage}_to_{stage}_robustness"
                entry["reason"] = segment_report.get("reason") or f"{stage.upper()} Cartesian path failed"
                return None, entry
            q_by_stage[stage] = q
            endpoint_reports[stage] = endpoint
            seed = q
            previous_stage = stage
        entry["ik_fk_endpoints"] = endpoint_reports
        entry["cartesian_local_path_robustness"] = {
            "safe": True,
            "segments": local_segment_reports,
        }

        entry["accepted"] = True
        accepted = dict(grasp)
        accepted["table_ik_fk_check"] = entry
        return accepted, entry
    except Exception as exc:
        entry["rejection_stage"] = "exception"
        entry["reason"] = f"{type(exc).__name__}: {exc}"
        if args.include_tracebacks:
            entry["traceback"] = traceback.format_exc()
        return None, entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline AnyGrasp table filter plus PyRoki IK/FK verification for FR3. No robot motion."
    )
    parser.add_argument("--grasps", required=True, type=Path, help="Input AnyGrasp grasps.json")
    parser.add_argument(
        "--calib-results",
        required=True,
        type=Path,
        help="Calibration result directory, robot_calib_result.npz, or calibration ZIP",
    )
    parser.add_argument(
        "--state-file",
        required=True,
        type=Path,
        help="Fresh FR3 state .npz containing current q (7 arm joints, or all actuated joints)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Accepted grasps JSON output")
    parser.add_argument("--report", type=Path, default=None, help="Detailed all-grasp report JSON")
    parser.add_argument("--description", default="fr3_description")
    parser.add_argument("--target-link", default="fr3_hand_tcp")
    parser.add_argument("--pose-frame", choices=("world", "base", "camera"), default="world")
    parser.add_argument(
        "--rot-offset-rpy",
        nargs=3,
        type=float,
        default=(0.0, math.pi / 2.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Execution orientation offset in radians; default is local +90 deg about Y",
    )
    parser.add_argument("--offset-mode", choices=("post", "pre", "none"), default="post")
    parser.add_argument("--table-z", type=float, default=0.0)
    parser.add_argument("--clearance-m", type=float, default=0.003)
    z_direction = parser.add_mutually_exclusive_group()
    z_direction.add_argument("--z-positive-down", dest="z_positive_down", action="store_true", help="Default: world +Z points down, so a point is above the table when z < --table-z (opposite of --z-positive-up).")
    z_direction.add_argument("--z-positive-up", dest="z_positive_down", action="store_false")
    parser.set_defaults(z_positive_down=True)
    center_filter = parser.add_mutually_exclusive_group()
    center_filter.add_argument(
        "--require-center-above-table", dest="require_center_above_table", action="store_true", help="Default: reject grasps whose center is at or below the table (opposite of --allow-center-at-or-below-table)."
    )
    center_filter.add_argument(
        "--allow-center-at-or-below-table", dest="require_center_above_table", action="store_false"
    )
    parser.set_defaults(require_center_above_table=True)
    parser.add_argument("--hover-z-offset", type=float, default=0.10, help="Added in FR3 base z, matching execution")
    parser.add_argument("--final-z-offset", type=float, default=0.0, help="Added in FR3 base z")
    parser.add_argument("--lift-z-offset", type=float, default=0.15, help="Added in FR3 base z, matching execution")
    parser.add_argument("--path-samples", type=int, default=25,
                        help="Dense Cartesian IK/FK samples for hover->final and final->lift.")
    parser.add_argument("--hover-path-linear-step-m", type=float, default=0.02,
                        help="Dense-IK validation spacing along the executed direct Cartesian path.")
    parser.add_argument("--hover-path-angular-step-deg", type=float, default=10.0,
                        help="Maximum orientation spacing for continuous hover IK waypoints.")
    parser.add_argument("--max-waypoint-joint-delta-rad", type=float, default=0.25,
                        help="Reject an IK branch jump larger than this between adjacent waypoints.")
    parser.add_argument("--robust-joint-margin-rad", type=float, default=0.20,
                        help="Reject candidates whose dense-IK path gets closer than this to a soft joint limit.")
    parser.add_argument("--max-home-hover-joint-delta-rad", type=float, default=1.40,
                        help="Reject candidates requiring excessive movement by any single joint.")
    parser.add_argument("--max-home-hover-total-joint-motion-rad", type=float, default=5.0,
                        help="Reject candidates with excessive summed dense-IK joint motion.")
    parser.add_argument("--max-home-hover-rotation-deg", type=float, default=100.0,
                        help="Reject candidates requiring excessive direct Cartesian orientation change.")
    parser.add_argument("--camera-extrinsic-dir", type=Path, default=None,
                        help="Optional directory of camera extrinsics for keep-out filtering. Every .npz directly in it whose name contains 'oak_', and every such .json without an .npz of the same name, gives one camera keep-out if it holds a camera-to-world transform.")
    parser.add_argument("--camera-keepout-radius-m", type=float, default=0.12,
                        help="Keep TCP/gripper sample points outside this radius around each camera optical center.")
    parser.add_argument("--safe-order", choices=("joint_motion", "input"), default="joint_motion",
                        help="Order accepted grasps by joint-path motion (default) or preserve input order.")
    parser.add_argument("--max-position-error-m", type=float, default=0.001)
    parser.add_argument("--max-orientation-error-deg", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=50.0)
    parser.add_argument("--ori-weight", type=float, default=10.0)
    parser.add_argument("--rest-weight", type=float, default=0.20,
                        help="Rest cost pulling each IK solve toward its warm-start seed; discourages IK branch jumps.")
    parser.add_argument("--open-finger-q", type=float, default=0.04)
    soft = parser.add_mutually_exclusive_group()
    soft.add_argument("--use-soft-limits", dest="use_soft_limits", action="store_true", help="Default: apply the soft joint-limit margins (opposite of --disable-soft-limits).")
    soft.add_argument("--disable-soft-limits", dest="use_soft_limits", action="store_false")
    parser.set_defaults(use_soft_limits=True)
    parser.add_argument("--include-tracebacks", action="store_true", help="Include per-grasp exception tracebacks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.path_samples < 2:
        raise ValueError("--path-samples must be at least 2")
    if args.clearance_m < 0.0:
        raise ValueError("--clearance-m must be non-negative")
    if args.hover_path_linear_step_m <= 0.0 or args.hover_path_angular_step_deg <= 0.0:
        raise ValueError("Hover path step sizes must be positive")
    if args.max_waypoint_joint_delta_rad <= 0.0:
        raise ValueError("--max-waypoint-joint-delta-rad must be positive")
    if args.robust_joint_margin_rad < 0.0:
        raise ValueError("--robust-joint-margin-rad must be non-negative")
    if args.max_home_hover_joint_delta_rad <= 0.0:
        raise ValueError("--max-home-hover-joint-delta-rad must be positive")
    if args.max_home_hover_total_joint_motion_rad <= 0.0:
        raise ValueError("--max-home-hover-total-joint-motion-rad must be positive")
    if args.max_home_hover_rotation_deg <= 0.0:
        raise ValueError("--max-home-hover-rotation-deg must be positive")
    if args.camera_keepout_radius_m < 0.0:
        raise ValueError("--camera-keepout-radius-m must be non-negative")

    grasps_path = as_path(args.grasps)
    output_path = as_path(args.out) if args.out else grasps_path.with_name("grasps_table_ik_safe.json")
    report_path = as_path(args.report) if args.report else grasps_path.with_name("grasps_table_ik_report.json")
    grasps = load_json(grasps_path)
    if not isinstance(grasps, list):
        raise TypeError("Input grasps JSON must be a list")

    calib_root, temporary = resolve_calib_root(args.calib_results)
    try:
        calibration = load_calibration(calib_root, args.pose_frame)
        calibration["camera_keepouts"] = load_camera_keepouts(args.camera_extrinsic_dir)
        current_state_q, state_key = load_current_arm_q(args.state_file)
        print("Loading FR3 and building PyRoki IK/FK problem...")
        context = build_pyroki_context(
            description=args.description,
            target_link=args.target_link,
            current_state_q=current_state_q,
            open_finger_q=args.open_finger_q,
            pos_weight=args.pos_weight,
            ori_weight=args.ori_weight,
            rest_weight=args.rest_weight,
        )
        current_hard = hard_limit_check(context, context.current_q)
        current_soft = soft_limit_check(context, context.current_q, args.use_soft_limits)
        if not current_hard["safe"]:
            raise RuntimeError(f"Current robot state violates URDF hard limits: {current_hard}")
        if not current_soft["safe"]:
            raise RuntimeError(
                f"Current robot state violates configured soft limits: {current_soft}. "
                "Move to a safer configuration or consciously use --disable-soft-limits."
            )

        print(f"Target link: {context.target_link} (index {context.target_link_index})")
        print(f"Actuated joints: {context.actuated_names}")
        print(f"Input grasps: {len(grasps)}")
        print(f"Camera keep-outs: {len(calibration['camera_keepouts'])} cameras, radius={args.camera_keepout_radius_m:.3f}m")
        R_offset = rpy_to_R(*args.rot_offset_rpy)
        accepted: List[Dict[str, Any]] = []
        entries: List[Dict[str, Any]] = []
        for index, grasp in enumerate(grasps):
            kept, entry = process_grasp(index, grasp, context, calibration, R_offset, args)
            entries.append(entry)
            status = "KEEP" if kept is not None else f"REJECT:{entry['rejection_stage']}"
            print(f"[{index + 1:03d}/{len(grasps):03d}] rank={entry['rank']} {status}")
            if kept is not None:
                accepted.append(kept)

        rejection_counts: Dict[str, int] = {}
        for entry in entries:
            if entry["accepted"]:
                continue
            stage = str(entry["rejection_stage"])
            rejection_counts[stage] = rejection_counts.get(stage, 0) + 1

        if args.safe_order == "joint_motion":
            accepted.sort(
                key=lambda grasp: (
                    float(
                        grasp["table_ik_fk_check"]["home_to_hover_cartesian_robustness"]
                        ["total_joint_motion_l1_rad"]
                    ),
                    -float(grasp.get("score", 0.0) or 0.0),
                )
            )

        report = {
            "summary": {
                "input_grasps": len(grasps),
                "accepted_grasps": len(accepted),
                "rejected_grasps": len(grasps) - len(accepted),
                "accepted_ranks": [entry["rank"] for entry in entries if entry["accepted"]],
                "accepted_output_ranks": [grasp.get("rank") for grasp in accepted],
                "safe_order": args.safe_order,
                "rejection_counts": rejection_counts,
                "no_robot_motion_commanded": True,
            },
            "configuration": {
                "grasps_path": str(grasps_path),
                "state_file": str(as_path(args.state_file)),
                "state_key": state_key,
                "robot_calib_path": calibration["robot_calib_path"],
                "fingertip_calib_path": calibration["fingertip_calib_path"],
                "description": args.description,
                "target_link": args.target_link,
                "pose_frame": args.pose_frame,
                "rot_offset_rpy_rad": list(args.rot_offset_rpy),
                "offset_mode": args.offset_mode,
                "table_z_m": args.table_z,
                "z_positive_down": args.z_positive_down,
                "require_center_above_table": args.require_center_above_table,
                "clearance_m": args.clearance_m,
                "hover_z_offset_base_m": args.hover_z_offset,
                "final_z_offset_base_m": args.final_z_offset,
                "lift_z_offset_base_m": args.lift_z_offset,
                "path_samples_per_segment": args.path_samples,
                "hover_path_linear_step_m": args.hover_path_linear_step_m,
                "hover_path_angular_step_deg": args.hover_path_angular_step_deg,
                "max_waypoint_joint_delta_rad": args.max_waypoint_joint_delta_rad,
                "robust_joint_margin_rad": args.robust_joint_margin_rad,
                "max_home_hover_joint_delta_rad": args.max_home_hover_joint_delta_rad,
                "max_home_hover_total_joint_motion_rad": args.max_home_hover_total_joint_motion_rad,
                "max_home_hover_rotation_deg": args.max_home_hover_rotation_deg,
                "camera_extrinsic_dir": str(as_path(args.camera_extrinsic_dir)) if args.camera_extrinsic_dir else None,
                "camera_keepout_radius_m": args.camera_keepout_radius_m,
                "camera_keepouts": calibration["camera_keepouts"],
                "safe_order": args.safe_order,
                "max_position_error_m": args.max_position_error_m,
                "max_orientation_error_deg": args.max_orientation_error_deg,
                "soft_limits_enabled": args.use_soft_limits,
                "soft_limits_rad": FR3_SOFT_LIMITS_RAD if args.use_soft_limits else None,
                "path_check_scope": (
                    "dense IK robustness evaluation of the executed direct Cartesian path plus sampled local checks; "
                    "PyRoki FK of fr3_hand_tcp; calibrated fingertip-vs-table only; "
                    "not full-link/self/environment collision checking"
                ),
            },
            "current_state": {
                "q_all_actuated_rad": context.current_q.tolist(),
                "hard_joint_limits": current_hard,
                "soft_joint_limits": current_soft,
            },
            "entries": entries,
        }
        save_json(output_path, accepted)
        save_json(report_path, report)

        print("\n========== RESULT ==========")
        print(f"Accepted: {len(accepted)}/{len(grasps)}")
        print(f"Rejection counts: {rejection_counts}")
        print(f"Safe grasps: {output_path}")
        print(f"Detailed report: {report_path}")
        print("No robot motion was commanded.")
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
