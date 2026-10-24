#!/usr/bin/env python3
r"""Robot/world (hand-eye) calibration with a gripper-held ChArUco board.

Input is a recording from aura/robot/gamepad_record_franka_3oak.py (pass its
--output-root directory as --session-root) in which the ChArUco board is rigidly
held in the gripper and moved through diverse poses in front of ONE fixed camera
that already has an extrinsic result from aura/calibration/Extract_extrinsic.py.
Each bundle_NNNNNN/meta.json carries robot.joint_angles_rad and
robot.sync_error_abs_ms; the image is read from
bundle_NNNNNN/<camera-name>/rgb.jpg. The older pose_xyzrpy /
capture_delay_from_anchor_ms meta fields are still accepted.

It computes T_base_from_ee by Franka/Panda-style modified-DH forward kinematics.
The estimated T_ee_from_board absorbs any constant flange/tool/board offset, so
T_world_from_base remains meaningful as long as the FK frame is rigidly attached
to the last robot link.

Usage (repository root, AURA_ENV conda env, after `set -a; source config/aura.env; set +a`):

    python aura/calibration/robot_world_calib_charuco.py \
        --session-root <recording dir> \
        --camera-name oak_<MXID> \
        --world-from-camera-file "$AURA_CALIB_ROOT/extrinsic_results/oak_<MXID>.npz" \
        --intrinsic-file "$AURA_CALIB_ROOT/intrinsic_results/1280x720/oak_<MXID>.npz" \
        --squares-x 4 --squares-y 5 --square-length 0.04 --marker-length 0.03 --dictionary DICT_4X4_50 \
        --output-dir "$AURA_CALIB_ROOT/robot_calibration_results"

Record the session with gamepad_record_franka_3oak.py --resolution high: its
default, low (640x400), does not match these 1280x720 intrinsics, and the
bundles are then dropped as drop_exception (Image size mismatch).

--camera-name defaults to $AURA_HANDEYE_OAK_ID and is required when that is
unset; it must be the bundle sub-directory name oak_<MXID> (with the oak_
prefix, unlike the bare MXIDs in AURA_OAK_IDS) and the camera of
--world-from-camera-file. A wrong name is not caught up front: every bundle is
then dropped as drop_exception. The authors' reference calibration was produced
with --max-robot-delay-ms 30 --prefer-joint-fk in addition (recorded in the
result JSON's "filters" block). Run with --diagnose-only first: it prints
per-status drop counts and writes scan_records_debug.json /
pose_filter_logs_debug.json without solving. It exits 0 even when every bundle
was dropped, so read the counts.

Outputs in --output-dir: robot_calib_result.npz (T_world_from_camera,
T_world_from_base, T_base_from_world, T_ee_from_board, T_board_from_ee), read by
aura/grasping/filter_grasps_table_collision_pyroki.py and the aura/robot/
executors, and robot_calib_result.json (summary and logs; it embeds the session,
extrinsic and image paths as passed, but not --intrinsic-file; do not publish it).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


DICT_MAP = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


# -----------------------------
# Basic SE(3) helper functions
# -----------------------------

def make_T(Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(Rm, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = Rm.T
    T_inv[:3, 3] = -Rm.T @ t
    return T_inv


def rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    Rm, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return make_T(Rm, np.asarray(tvec, dtype=np.float64).reshape(3))


def params_to_T(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64).reshape(6)
    t = params[:3]
    rotvec = params[3:]
    Rm = R.from_rotvec(rotvec).as_matrix()
    return make_T(Rm, t)


def T_to_params(T: np.ndarray) -> np.ndarray:
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    t = T[:3, 3]
    return np.concatenate([t, rotvec], axis=0)


def se3_residual(T_err: np.ndarray, rot_weight: float = 1.0, trans_weight: float = 1.0) -> np.ndarray:
    trans = trans_weight * T_err[:3, 3]
    rot = rot_weight * R.from_matrix(T_err[:3, :3]).as_rotvec()
    return np.concatenate([trans, rot], axis=0)


def quat_average(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """Average rotations by quaternion eigenvector method."""
    if not rotations:
        return np.eye(3, dtype=np.float64)
    quats = R.from_matrix(np.stack(rotations, axis=0)).as_quat()  # xyzw
    ref = quats[0]
    aligned = []
    for q in quats:
        if np.dot(q, ref) < 0:
            q = -q
        aligned.append(q)
    aligned = np.asarray(aligned, dtype=np.float64)
    M = aligned.T @ aligned
    eigvals, eigvecs = np.linalg.eigh(M)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    q_avg /= np.linalg.norm(q_avg)
    return R.from_quat(q_avg).as_matrix()


def average_transforms(Ts: Sequence[np.ndarray]) -> np.ndarray:
    if not Ts:
        raise ValueError("No transforms to average.")
    R_avg = quat_average([T[:3, :3] for T in Ts])
    t_avg = np.mean(np.stack([T[:3, 3] for T in Ts], axis=0), axis=0)
    return make_T(R_avg, t_avg)


def rotation_angle_deg(Rm: np.ndarray) -> float:
    return float(np.rad2deg(np.linalg.norm(R.from_matrix(Rm).as_rotvec())))


# -----------------------------
# IO helpers
# -----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_intrinsics_from_session(session_root: Path, camera_name: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Load intrinsics from a calibration_manifest in session_summary.json or recording_summary.json.

    Only execute_anygrasp_franka_record_lift.py writes a calibration_manifest (into
    both summaries); gamepad_record_franka_3oak.py and record_3oak.py recordings
    have none, so for hand-eye sessions pass --intrinsic-file explicitly.
    """
    checked = []
    for summary_name in ["session_summary.json", "recording_summary.json"]:
        summary_path = session_root / summary_name
        checked.append(str(summary_path))
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        manifest = summary.get("calibration_manifest")
        if manifest is None:
            continue
        if camera_name not in manifest:
            raise KeyError(f"Camera {camera_name!r} not found in {summary_name} calibration_manifest.")
        info = manifest[camera_name]
        K = np.asarray(info["K"], dtype=np.float64)
        dist = np.asarray(info["dist"], dtype=np.float64).reshape(-1)
        size = (int(info["image_width"]), int(info["image_height"]))
        return K, dist, size
    raise FileNotFoundError(
        "Could not load intrinsics from session summaries. "
        "Your current memory-first recording likely has no calibration_manifest. "
        "Please pass --intrinsic-file $AURA_CALIB_ROOT/intrinsic_results/<WxH>/oak_<MXID>.npz from Extract_intrinsic.py. Checked: "
        + ", ".join(checked)
    )


def load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        K = np.asarray(data["K"], dtype=np.float64)
        dist = np.asarray(data["dist"], dtype=np.float64).reshape(-1)
        image_size = tuple(int(x) for x in np.asarray(data["image_size"]).reshape(-1))
        return K, dist, (image_size[0], image_size[1])
    if path.suffix.lower() == ".json":
        data = load_json(path)
        K = np.asarray(data["K"], dtype=np.float64)
        dist = np.asarray(data["dist"], dtype=np.float64).reshape(-1)
        image_size = tuple(int(x) for x in data["image_size"])
        return K, dist, (image_size[0], image_size[1])
    raise ValueError(f"Unsupported intrinsic file format: {path}")


def _matrix_from_any(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return None
    try:
        arr = np.asarray(obj, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape == (4, 4):
        return arr
    return None


def load_world_from_camera(path: Path, camera_name: Optional[str] = None) -> np.ndarray:
    """
    Supports several formats:
    - JSON written by Extract_extrinsic.py: T_world_from_cam / phys_twin_c2w / T_board_from_cam
    - NPZ with the same keys
    - Combined JSON mapping camera names to nested dicts
    - Raw 4x4 JSON array
    """
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        for key in ["T_world_from_cam", "T_world_from_camera", "phys_twin_c2w", "c2w", "T_board_from_cam"]:
            if key in data:
                T = _matrix_from_any(data[key])
                if T is not None:
                    return T
        raise KeyError(f"Could not find a 4x4 world-from-camera matrix in NPZ: {path}")

    data = load_json(path)
    # Raw 4x4 array
    T = _matrix_from_any(data)
    if T is not None:
        return T

    # Direct keys
    for key in ["T_world_from_cam", "T_world_from_camera", "phys_twin_c2w", "c2w", "T_board_from_cam"]:
        if key in data:
            T = _matrix_from_any(data[key])
            if T is not None:
                return T

    # Nested mapping by camera name
    if camera_name is not None and camera_name in data:
        nested = data[camera_name]
        if isinstance(nested, dict):
            for key in ["T_world_from_cam", "T_world_from_camera", "phys_twin_c2w", "c2w", "T_board_from_cam"]:
                if key in nested:
                    T = _matrix_from_any(nested[key])
                    if T is not None:
                        return T
        else:
            T = _matrix_from_any(nested)
            if T is not None:
                return T

    raise KeyError(f"Could not parse T_world_from_cam from {path}")


def find_image(bundle_dir: Path, camera_name: str) -> Path:
    for fn in ["rgb.png", "rgb.jpg", "rgb.jpeg"]:
        p = bundle_dir / camera_name / fn
        if p.exists():
            return p
    raise FileNotFoundError(f"No RGB image found under {bundle_dir / camera_name}")


# -----------------------------
# ChArUco detection
# -----------------------------

def make_charuco_pnp_points(board, charuco_corners: np.ndarray, charuco_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build correct 3D-2D pairs for interpolated ChArUco corners.

    Do NOT use board.matchImagePoints(charuco_corners, charuco_ids) here.
    On OpenCV 4.7+ that API is easy to misuse with interpolated ChArUco
    chessboard corners and can produce wrong correspondences.  This function
    follows the same method as Extract_extrinsic.py:

        image point  = charuco_corners[i]
        object point = board.getChessboardCorners()[charuco_ids[i]]
    """
    if charuco_corners is None or charuco_ids is None:
        raise RuntimeError("No ChArUco corners/ids available for PnP.")
    if not hasattr(board, "getChessboardCorners"):
        raise RuntimeError("This OpenCV CharucoBoard does not expose getChessboardCorners().")

    ids = np.asarray(charuco_ids).reshape(-1).astype(np.int32)
    all_obj = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    if ids.size == 0:
        raise RuntimeError("No ChArUco ids available for PnP.")
    if np.any(ids < 0) or np.any(ids >= len(all_obj)):
        raise RuntimeError(
            f"ChArUco ids out of board range: min={ids.min()}, max={ids.max()}, "
            f"board corners={len(all_obj)}."
        )

    obj_points = all_obj[ids].reshape(-1, 3).astype(np.float32)
    img_points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
    if len(obj_points) != len(img_points):
        raise RuntimeError(
            f"Object/image point length mismatch: obj={len(obj_points)}, img={len(img_points)}."
        )
    return obj_points, img_points


def compute_reproj_rms_from_points(obj_points: np.ndarray, img_points: np.ndarray,
                                   rvec: np.ndarray, tvec: np.ndarray,
                                   K: np.ndarray, dist: np.ndarray) -> Optional[float]:
    if obj_points is None or img_points is None:
        return None
    obj = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img_points, dtype=np.float64).reshape(-1, 2)
    if len(obj) == 0 or len(obj) != len(img):
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))


def detect_and_estimate_pose(img_bgr: np.ndarray, board, detector,
                             K: np.ndarray, dist: np.ndarray,
                             min_markers: int, min_charuco: int):
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img_bgr)
    num_markers = 0 if marker_ids is None else len(marker_ids)
    num_charuco = 0 if charuco_ids is None else len(charuco_ids)

    if num_markers < min_markers or num_charuco < min_charuco:
        raise RuntimeError(
            f"Not enough detections: markers={num_markers}, charuco={num_charuco}; "
            f"need markers>={min_markers}, charuco>={min_charuco}."
        )

    obj_points, img_points = make_charuco_pnp_points(board, charuco_corners, charuco_ids)
    if len(obj_points) < min_charuco:
        raise RuntimeError(
            f"Only {len(obj_points)} ChArUco object/image points, need at least {min_charuco}."
        )

    # ChArUco board is planar; IPPE is a good first choice when available.
    pnp_flag = cv2.SOLVEPNP_IPPE if hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_points,
        imagePoints=img_points,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=pnp_flag,
    )

    # Fallback to iterative PnP in case IPPE fails for a marginal view.
    if not success or rvec is None or tvec is None:
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_points,
            imagePoints=img_points,
            cameraMatrix=K,
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not success or rvec is None or tvec is None:
        raise RuntimeError("Pose estimation failed even though ChArUco detection succeeded.")

    rms = compute_reproj_rms_from_points(obj_points, img_points, rvec, tvec, K, dist)
    return {
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "num_markers": int(num_markers),
        "num_charuco": int(num_charuco),
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3),
        "rms": rms,
        "pnp_method": "manual_charuco_id_to_chessboard_corner",
    }


def draw_overlay(img_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray,
                 det: Dict[str, Any], axis_length: float, footer: Optional[str] = None) -> np.ndarray:
    out = img_bgr.copy()
    if det["marker_ids"] is not None and len(det["marker_ids"]) > 0:
        cv2.aruco.drawDetectedMarkers(out, det["marker_corners"], det["marker_ids"])
    if det["charuco_ids"] is not None and len(det["charuco_ids"]) > 0:
        cv2.aruco.drawDetectedCornersCharuco(out, det["charuco_corners"], det["charuco_ids"])
    cv2.drawFrameAxes(out, K, dist, det["rvec"].reshape(3, 1), det["tvec"].reshape(3, 1), axis_length)

    lines = [
        f"markers={det['num_markers']}",
        f"charuco={det['num_charuco']}",
        f"rms={det['rms']:.4f}" if det["rms"] is not None else "rms=n/a",
    ]
    if footer:
        lines.append(footer)

    y = 24
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
        y += 24
    return out


# -----------------------------
# Robot pose helpers
# -----------------------------

def pose_xyzrpy_to_T(pose_xyzrpy: Sequence[float], euler_seq: str = "xyz", euler_degrees: bool = False) -> np.ndarray:
    pose_xyzrpy = np.asarray(pose_xyzrpy, dtype=np.float64).reshape(6)
    xyz = pose_xyzrpy[:3]
    rpy = pose_xyzrpy[3:]
    Rm = R.from_euler(euler_seq, rpy, degrees=euler_degrees).as_matrix()
    return make_T(Rm, xyz)


def pose_distance(T_a: np.ndarray, T_b: np.ndarray) -> Tuple[float, float]:
    dT = invert_T(T_a) @ T_b
    pos = float(np.linalg.norm(dT[:3, 3]))
    ang = rotation_angle_deg(dT[:3, :3])
    return pos, ang


# -----------------------------
# Franka FK from joint angles
# -----------------------------
def mdh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Modified DH transform: Rot_x(alpha) Trans_x(a) Rot_z(theta) Trans_z(d)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def franka_fk_mdh(joint_angles_rad: Sequence[float], extra_ee_z_m: float = 0.0) -> np.ndarray:
    """Approximate Franka Panda/FR3 modified-DH FK from base to last EE frame.

    The FR3 and Panda have the same 7-DOF serial kinematic structure for this
    calibration purpose. This frame does not have to be exactly the same as the
    libfranka configured TCP: the solver estimates a constant T_ee_from_board,
    so any fixed flange/TCP offset is absorbed there.
    """
    q = np.asarray(joint_angles_rad, dtype=np.float64).reshape(-1)
    if q.size != 7:
        raise ValueError(f"Expected 7 Franka joint angles, got {q.size}.")

    # Modified DH parameters commonly used for Franka Emika Panda/FR3 geometry.
    # Rows: (a_{i-1}, d_i, alpha_{i-1}, theta_i).
    a = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
    d = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]
    alpha = [0.0, -math.pi / 2.0, math.pi / 2.0, math.pi / 2.0,
             -math.pi / 2.0, math.pi / 2.0, math.pi / 2.0]

    T = np.eye(4, dtype=np.float64)
    for ai, di, al, qi in zip(a, d, alpha, q):
        T = T @ mdh_transform(ai, di, al, float(qi))

    if abs(extra_ee_z_m) > 0.0:
        T = T @ make_T(np.eye(3), np.array([0.0, 0.0, float(extra_ee_z_m)]))
    return T


def robot_meta_to_T_base_from_ee(robot: Dict[str, Any], args) -> Tuple[np.ndarray, str]:
    """Read robot pose from meta. Prefer joint FK for current joint-only recordings."""
    if args.prefer_joint_fk and "joint_angles_rad" in robot:
        return franka_fk_mdh(robot["joint_angles_rad"], extra_ee_z_m=args.extra_ee_z_m), "joint_fk"

    if "pose_xyzrpy" in robot:
        return pose_xyzrpy_to_T(
            robot["pose_xyzrpy"],
            euler_seq=args.robot_euler_seq,
            euler_degrees=args.robot_euler_degrees,
        ), "pose_xyzrpy"

    if "joint_angles_rad" in robot:
        return franka_fk_mdh(robot["joint_angles_rad"], extra_ee_z_m=args.extra_ee_z_m), "joint_fk"

    raise KeyError("Robot meta has neither pose_xyzrpy nor joint_angles_rad.")


def robot_meta_sync_error(robot: Dict[str, Any]) -> Tuple[float, float, str]:
    """Return (abs_error_ms, signed_error_ms, source_key) across old/new schemas."""
    if "sync_error_abs_ms" in robot:
        abs_ms = float(robot["sync_error_abs_ms"])
        if "sync_error_from_target_ms" in robot:
            signed_ms = float(robot["sync_error_from_target_ms"])
        elif "sync_error_from_anchor_ms" in robot:
            signed_ms = float(robot["sync_error_from_anchor_ms"])
        else:
            signed_ms = abs_ms
        return abs_ms, signed_ms, "sync_error_abs_ms"

    if "capture_delay_from_anchor_ms" in robot:
        signed_ms = float(robot["capture_delay_from_anchor_ms"])
        return abs(signed_ms), signed_ms, "capture_delay_from_anchor_ms"

    if "host_ts" in robot and "sync_target_ts" in robot:
        signed_ms = 1000.0 * (float(robot["host_ts"]) - float(robot["sync_target_ts"]))
        return abs(signed_ms), signed_ms, "host_ts_minus_sync_target_ts"

    raise KeyError("Could not find robot sync error field in meta robot dict.")


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class Sample:
    bundle_name: str
    bundle_index: int
    image_path: Path
    robot_delay_ms: float
    camera_host_delta_ms: float
    camera_match_delta_ms: float
    robot_host_ts: float
    T_base_from_ee: np.ndarray
    T_world_from_board_vision: np.ndarray
    det: Dict[str, Any]
    meta: Dict[str, Any]
    selected_reason: str = ""


# -----------------------------
# Bundle loading and filtering
# -----------------------------

def load_candidate_samples(
    session_root: Path,
    camera_name: str,
    T_world_from_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    image_size: Tuple[int, int],
    board,
    detector,
    args,
) -> Tuple[List[Sample], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    kept: List[Sample] = []
    axis_length = args.axis_length if args.axis_length is not None else (args.square_length * 2.0)

    bundle_dirs = sorted([p for p in session_root.iterdir() if p.is_dir() and p.name.startswith("bundle_")])
    if args.bundle_start_index is not None:
        bundle_dirs = [p for p in bundle_dirs if int(p.name.split('_')[-1]) >= args.bundle_start_index]
    if args.bundle_end_index is not None:
        bundle_dirs = [p for p in bundle_dirs if int(p.name.split('_')[-1]) <= args.bundle_end_index]
    if args.max_bundles is not None:
        bundle_dirs = bundle_dirs[:args.max_bundles]

    last_seen_robot_ts: Optional[float] = None
    for bundle_dir in bundle_dirs:
        meta_path = bundle_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = load_json(meta_path)
        record: Dict[str, Any] = {
            "bundle_name": bundle_dir.name,
            "bundle_index": int(meta.get("bundle_index", -1)),
            "status": "unknown",
        }

        try:
            robot = meta["robot"]
            sources = meta["sources"]
            if camera_name not in sources:
                raise KeyError(f"Camera {camera_name!r} missing in meta sources.")

            # New memory-first bundles may not have anchor_source. For camera-specific
            # calibration, using the selected camera as anchor makes deltas zero while
            # match_span_ms still checks cross-camera sync quality.
            anchor_name = meta.get("anchor_source", camera_name)
            if anchor_name not in sources:
                anchor_name = camera_name

            robot_abs_delay_ms, robot_signed_delay_ms, robot_delay_source = robot_meta_sync_error(robot)
            if robot_abs_delay_ms > args.max_robot_delay_ms:
                record.update({
                    "status": "drop_robot_delay",
                    "robot_delay_ms": robot_signed_delay_ms,
                    "robot_abs_delay_ms": robot_abs_delay_ms,
                    "robot_delay_source": robot_delay_source,
                })
                records.append(record)
                continue

            match_span_ms = meta.get("match_span_ms", None)
            if match_span_ms is not None and float(match_span_ms) > args.max_match_span_ms:
                record.update({
                    "status": "drop_match_span",
                    "match_span_ms": float(match_span_ms),
                    "robot_abs_delay_ms": robot_abs_delay_ms,
                })
                records.append(record)
                continue

            robot_host_ts = float(robot["host_ts"])
            if args.drop_duplicate_robot_timestamps and last_seen_robot_ts is not None and abs(robot_host_ts - last_seen_robot_ts) < 1e-9:
                record.update({
                    "status": "drop_duplicate_robot_ts",
                    "robot_delay_ms": robot_signed_delay_ms,
                    "robot_abs_delay_ms": robot_abs_delay_ms,
                    "robot_host_ts": robot_host_ts,
                })
                records.append(record)
                continue

            cam_info = sources[camera_name]
            anchor_info = sources[anchor_name]
            camera_host_delta_ms = 1000.0 * (float(cam_info["host_ts"]) - float(anchor_info["host_ts"]))
            camera_match_delta_ms = 1000.0 * (float(cam_info["match_ts"]) - float(anchor_info["match_ts"]))

            if abs(camera_host_delta_ms) > args.max_camera_host_delta_ms:
                record.update({
                    "status": "drop_camera_host_delta",
                    "robot_delay_ms": robot_signed_delay_ms,
                    "robot_abs_delay_ms": robot_abs_delay_ms,
                    "camera_host_delta_ms": camera_host_delta_ms,
                })
                records.append(record)
                continue

            if abs(camera_match_delta_ms) > args.max_camera_match_delta_ms:
                record.update({
                    "status": "drop_camera_match_delta",
                    "robot_delay_ms": robot_signed_delay_ms,
                    "robot_abs_delay_ms": robot_abs_delay_ms,
                    "camera_match_delta_ms": camera_match_delta_ms,
                })
                records.append(record)
                continue

            image_path = find_image(bundle_dir, camera_name)
            img = cv2.imread(str(image_path))
            if img is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            h, w = img.shape[:2]
            if (not getattr(args, "ignore_image_size_check", False)) and (w, h) != image_size:
                raise RuntimeError(f"Image size mismatch for {image_path}: got {(w, h)}, expected {image_size}. Pass an --intrinsic-file fitted at the size you got, or record again at the expected size (gamepad_record_franka_3oak.py defaults to low, 640x400; --resolution high is 1280x720). Use --ignore-image-size-check only if your NPZ stores image_size as (H,W) but K matches this recording.")

            det = detect_and_estimate_pose(
                img_bgr=img,
                board=board,
                detector=detector,
                K=K,
                dist=dist,
                min_markers=args.min_markers,
                min_charuco=args.min_charuco,
            )
            if args.max_reproj_rms >= 0 and det["rms"] is not None and det["rms"] > args.max_reproj_rms:
                record.update({
                    "status": "drop_reproj_rms",
                    "robot_delay_ms": robot_signed_delay_ms,
                    "robot_abs_delay_ms": robot_abs_delay_ms,
                    "reproj_rms": float(det["rms"]),
                })
                records.append(record)
                continue

            T_cam_from_board = rt_to_T(det["rvec"], det["tvec"])
            T_world_from_board_vision = T_world_from_cam @ T_cam_from_board
            T_base_from_ee, robot_pose_source = robot_meta_to_T_base_from_ee(robot, args)

            sample = Sample(
                bundle_name=bundle_dir.name,
                bundle_index=int(meta.get("bundle_index", -1)),
                image_path=image_path,
                robot_delay_ms=robot_signed_delay_ms,
                camera_host_delta_ms=float(camera_host_delta_ms),
                camera_match_delta_ms=float(camera_match_delta_ms),
                robot_host_ts=robot_host_ts,
                T_base_from_ee=T_base_from_ee,
                T_world_from_board_vision=T_world_from_board_vision,
                det=det,
                meta=meta,
            )
            kept.append(sample)
            record.update({
                "status": "keep_pre_pose_filter",
                "robot_delay_ms": robot_signed_delay_ms,
                "robot_abs_delay_ms": robot_abs_delay_ms,
                "robot_delay_source": robot_delay_source,
                "robot_pose_source": robot_pose_source,
                "camera_host_delta_ms": camera_host_delta_ms,
                "camera_match_delta_ms": camera_match_delta_ms,
                "match_span_ms": match_span_ms,
                "reproj_rms": det["rms"],
                "num_markers": det["num_markers"],
                "num_charuco": det["num_charuco"],
            })
            if args.save_detection_overlays:
                out_dir = Path(args.output_dir) / "detection_overlays_all"
                out_dir.mkdir(parents=True, exist_ok=True)
                overlay = draw_overlay(
                    img, K, dist, det, axis_length,
                    footer=f"robot_abs_sync={robot_abs_delay_ms:.1f} ms src={robot_pose_source}",
                )
                cv2.imwrite(str(out_dir / f"{bundle_dir.name}_{camera_name}.png"), overlay)
            last_seen_robot_ts = robot_host_ts
        except Exception as exc:
            record.update({"status": "drop_exception", "error": str(exc)})
        records.append(record)

    kept.sort(key=lambda s: s.bundle_index)
    return kept, records

def apply_pose_diversity_filter(samples: Sequence[Sample], min_pos_m: float, min_rot_deg: float,
                                max_pos_step_m: Optional[float], max_rot_step_deg: Optional[float]) -> Tuple[List[Sample], List[Dict[str, Any]]]:
    selected: List[Sample] = []
    logs: List[Dict[str, Any]] = []

    last_kept: Optional[Sample] = None
    for s in samples:
        if last_kept is None:
            s.selected_reason = "first_sample"
            selected.append(s)
            last_kept = s
            logs.append({"bundle_name": s.bundle_name, "status": "keep_pose_filter", "reason": s.selected_reason})
            continue

        pos_diff, rot_diff = pose_distance(last_kept.T_base_from_ee, s.T_base_from_ee)

        if max_pos_step_m is not None and pos_diff > max_pos_step_m:
            logs.append({
                "bundle_name": s.bundle_name,
                "status": "drop_pose_jump",
                "pos_diff_m": pos_diff,
                "rot_diff_deg": rot_diff,
            })
            continue
        if max_rot_step_deg is not None and rot_diff > max_rot_step_deg:
            logs.append({
                "bundle_name": s.bundle_name,
                "status": "drop_rot_jump",
                "pos_diff_m": pos_diff,
                "rot_diff_deg": rot_diff,
            })
            continue

        if pos_diff < min_pos_m and rot_diff < min_rot_deg:
            logs.append({
                "bundle_name": s.bundle_name,
                "status": "drop_too_similar",
                "pos_diff_m": pos_diff,
                "rot_diff_deg": rot_diff,
            })
            continue

        s.selected_reason = f"pos_diff={pos_diff:.4f}m rot_diff={rot_diff:.2f}deg"
        selected.append(s)
        last_kept = s
        logs.append({"bundle_name": s.bundle_name, "status": "keep_pose_filter", "reason": s.selected_reason})

    return selected, logs


# -----------------------------
# Calibration solve
# -----------------------------

def make_motion_pairs(samples: Sequence[Sample], max_pairs: int) -> List[Tuple[int, int]]:
    n = len(samples)
    all_pairs = list(itertools.combinations(range(n), 2))
    if len(all_pairs) <= max_pairs:
        return all_pairs
    idxs = np.linspace(0, len(all_pairs) - 1, max_pairs, dtype=int)
    return [all_pairs[i] for i in idxs]


def estimate_T_ee_from_board_via_relative_motion(samples: Sequence[Sample], rot_weight: float, trans_weight: float,
                                                 max_pairs: int, init_T: Optional[np.ndarray] = None) -> np.ndarray:
    if len(samples) < 3:
        raise ValueError("Need at least 3 samples to estimate T_ee_from_board.")
    pairs = make_motion_pairs(samples, max_pairs=max_pairs)

    def residuals(x: np.ndarray) -> np.ndarray:
        X = params_to_T(x)
        res = []
        for i, j in pairs:
            A = invert_T(samples[i].T_base_from_ee) @ samples[j].T_base_from_ee
            B = invert_T(samples[i].T_world_from_board_vision) @ samples[j].T_world_from_board_vision
            # A X = X B
            T_err = invert_T(A @ X) @ (X @ B)
            res.append(se3_residual(T_err, rot_weight=rot_weight, trans_weight=trans_weight))
        return np.concatenate(res, axis=0)

    x0 = T_to_params(np.eye(4) if init_T is None else init_T)
    result = least_squares(residuals, x0=x0, method="trf", loss="soft_l1", f_scale=0.05, max_nfev=2000)
    return params_to_T(result.x)


def estimate_T_world_from_base_given_X(samples: Sequence[Sample], T_ee_from_board: np.ndarray) -> np.ndarray:
    T_candidates = []
    X_inv = invert_T(T_ee_from_board)
    for s in samples:
        # W_B = W_board * X^{-1} * (B_E)^{-1}
        T_world_from_base_i = s.T_world_from_board_vision @ X_inv @ invert_T(s.T_base_from_ee)
        T_candidates.append(T_world_from_base_i)
    return average_transforms(T_candidates)


def joint_refine(samples: Sequence[Sample], T_world_from_base_init: np.ndarray, T_ee_from_board_init: np.ndarray,
                 rot_weight: float, trans_weight: float) -> Tuple[np.ndarray, np.ndarray, Any]:
    x0 = np.concatenate([T_to_params(T_world_from_base_init), T_to_params(T_ee_from_board_init)], axis=0)

    def residuals(x: np.ndarray) -> np.ndarray:
        T_world_from_base = params_to_T(x[:6])
        T_ee_from_board = params_to_T(x[6:])
        res = []
        for s in samples:
            pred = T_world_from_base @ s.T_base_from_ee @ T_ee_from_board
            T_err = invert_T(s.T_world_from_board_vision) @ pred
            res.append(se3_residual(T_err, rot_weight=rot_weight, trans_weight=trans_weight))
        return np.concatenate(res, axis=0)

    result = least_squares(residuals, x0=x0, method="trf", loss="soft_l1", f_scale=0.05, max_nfev=4000)
    return params_to_T(result.x[:6]), params_to_T(result.x[6:]), result


def compute_per_sample_errors(samples: Sequence[Sample], T_world_from_base: np.ndarray, T_ee_from_board: np.ndarray) -> List[Dict[str, Any]]:
    out = []
    for s in samples:
        pred = T_world_from_base @ s.T_base_from_ee @ T_ee_from_board
        T_err = invert_T(s.T_world_from_board_vision) @ pred
        pos_err_m = float(np.linalg.norm(T_err[:3, 3]))
        rot_err_deg = rotation_angle_deg(T_err[:3, :3])
        out.append({
            "bundle_name": s.bundle_name,
            "bundle_index": s.bundle_index,
            "image_path": str(s.image_path),
            "robot_delay_ms": s.robot_delay_ms,
            "camera_host_delta_ms": s.camera_host_delta_ms,
            "camera_match_delta_ms": s.camera_match_delta_ms,
            "num_markers": s.det["num_markers"],
            "num_charuco": s.det["num_charuco"],
            "reproj_rms": s.det["rms"],
            "residual_translation_m": pos_err_m,
            "residual_rotation_deg": rot_err_deg,
            "selected_reason": s.selected_reason,
        })
    return out


def reject_outliers(samples: Sequence[Sample], errors: Sequence[Dict[str, Any]],
                    max_trans_err_m: Optional[float], max_rot_err_deg: Optional[float]) -> Tuple[List[Sample], List[Dict[str, Any]]]:
    kept = []
    logs = []
    for s, e in zip(samples, errors):
        if max_trans_err_m is not None and e["residual_translation_m"] > max_trans_err_m:
            logs.append({"bundle_name": s.bundle_name, "status": "drop_final_trans_outlier", "trans_err_m": e["residual_translation_m"]})
            continue
        if max_rot_err_deg is not None and e["residual_rotation_deg"] > max_rot_err_deg:
            logs.append({"bundle_name": s.bundle_name, "status": "drop_final_rot_outlier", "rot_err_deg": e["residual_rotation_deg"]})
            continue
        kept.append(s)
        logs.append({"bundle_name": s.bundle_name, "status": "keep_final_inlier"})
    return kept, logs


# -----------------------------
# Main
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robot calibration with a gripper-held ChArUco board and a fixed external camera. Supports joint-only memory-first recordings.")
    p.add_argument("--session-root", type=Path, required=True, help="Recording directory written by aura/robot/gamepad_record_franka_3oak.py (its --output-root; contains bundle_*/meta.json)")
    p.add_argument("--camera-name", type=str, default=os.environ.get("AURA_HANDEYE_OAK_ID") or None,
                   help="Main camera used for ChArUco detection: the bundle sub-directory name oak_<MXID>, same camera as --world-from-camera-file. Defaults to $AURA_HANDEYE_OAK_ID (also oak_<MXID>, with the prefix, unlike the bare MXIDs in AURA_OAK_IDS); required when that is unset.")
    p.add_argument("--world-from-camera-file", type=Path, required=True,
                   help="JSON/NPZ containing T_world_from_cam for the main camera")
    p.add_argument("--intrinsic-file", type=Path, default=None,
                   help=".npz from Extract_intrinsic.py for --camera-name at the recording resolution, e.g. $AURA_CALIB_ROOT/intrinsic_results/1280x720/oak_<MXID>.npz (a .json is read only if it has K, dist and image_size; the Extract_intrinsic.py .json report has no image_size). Pass it: the fallback (a calibration_manifest in session_summary.json / recording_summary.json) is never written by gamepad_record_franka_3oak.py.")
    p.add_argument("--output-dir", type=Path, required=True)

    # Robot pose interpretation / joint-only FK
    p.add_argument("--robot-euler-seq", type=str, default="xyz",
                   help="Euler sequence used by old pose_xyzrpy recordings. Default: xyz.")
    p.add_argument("--robot-euler-degrees", action="store_true", help="Interpret old pose_xyzrpy angles in degrees instead of radians")
    p.add_argument("--prefer-joint-fk", action="store_true",
                   help="If both pose_xyzrpy and joint_angles_rad exist, prefer FK from joint_angles_rad.")
    p.add_argument("--extra-ee-z-m", type=float, default=0.0,
                   help="Optional fixed extra translation along FK EE z axis. Usually leave 0; T_ee_from_board absorbs fixed tool offsets.")

    # Board
    p.add_argument("--squares-x", type=int, required=True)
    p.add_argument("--squares-y", type=int, required=True)
    p.add_argument("--square-length", type=float, required=True)
    p.add_argument("--marker-length", type=float, required=True)
    p.add_argument("--dictionary", type=str, default="DICT_4X4_50", choices=sorted(DICT_MAP.keys()))
    p.add_argument("--min-markers", type=int, default=3)
    p.add_argument("--min-charuco", type=int, default=8)
    p.add_argument("--max-reproj-rms", type=float, default=1.5,
                   help="Drop detections whose ChArUco reprojection RMS exceeds this. Set negative to disable this filter for diagnosis.")
    p.add_argument("--ignore-image-size-check", action="store_true",
                   help="Do not drop images whose size differs from the intrinsic image_size. Use only if the NPZ stores image_size as (H,W) while K is for this recording's resolution; for a session recorded at another resolution, pass intrinsics fitted at that resolution instead.")
    p.add_argument("--diagnose-only", action="store_true",
                   help="Scan bundles, print/save drop counts, then exit before solving.")
    p.add_argument("--axis-length", type=float, default=None)

    # Delay / sync filtering
    p.add_argument("--max-robot-delay-ms", type=float, default=80.0,
                   help="Drop bundle if robot sync error exceeds this. Supports old capture_delay_from_anchor_ms and new sync_error_abs_ms.")
    p.add_argument("--max-match-span-ms", type=float, default=45.0,
                   help="Drop bundle if meta.match_span_ms exceeds this; useful for current offline-synced multi-OAK recordings.")
    p.add_argument("--max-camera-host-delta-ms", type=float, default=80.0,
                   help="Drop bundle if abs(main camera host_ts - anchor host_ts) exceeds this.")
    p.add_argument("--max-camera-match-delta-ms", type=float, default=80.0,
                   help="Drop bundle if abs(main camera match_ts - anchor match_ts) exceeds this.")
    p.add_argument("--drop-duplicate-robot-timestamps", action="store_true",
                   help="Drop bundle if robot.host_ts is exactly the same as the previous kept-pre-pose-filter sample.")

    # Pose diversity filtering
    p.add_argument("--min-position-diff-m", type=float, default=0.02,
                   help="Greedy pose diversity: require at least this translation OR the min rotation diff.")
    p.add_argument("--min-rotation-diff-deg", type=float, default=5.0,
                   help="Greedy pose diversity: require at least this rotation OR the min translation diff.")
    p.add_argument("--max-position-step-m", type=float, default=None,
                   help="Optional: drop sample if translation jump from last kept sample exceeds this.")
    p.add_argument("--max-rotation-step-deg", type=float, default=None,
                   help="Optional: drop sample if rotation jump from last kept sample exceeds this.")

    # Solver behavior
    p.add_argument("--motion-max-pairs", type=int, default=800)
    p.add_argument("--trans-weight", type=float, default=1.0)
    p.add_argument("--rot-weight", type=float, default=1.0)
    p.add_argument("--init-ee-board-file", type=Path, default=None,
                   help="Optional JSON with the initial T_ee_from_board, as a raw 4x4 array or {\"T_ee_from_board\": [[...]]}. Do not pass an NPZ or a file that also holds a camera pose, such as robot_calib_result.npz/.json: the camera pose is read instead, and an NPZ without one raises KeyError.")
    p.add_argument("--init-world-base-file", type=Path, default=None,
                   help="Optional JSON with the initial T_world_from_base, as a raw 4x4 array or {\"T_world_from_base\": [[...]]}. Same limits as --init-ee-board-file.")

    # Final outlier rejection and rerun
    p.add_argument("--final-max-trans-err-m", type=float, default=0.03,
                   help="After first solve, drop samples with translation residual above this and rerun. Set negative to disable.")
    p.add_argument("--final-max-rot-err-deg", type=float, default=5.0,
                   help="After first solve, drop samples with rotation residual above this and rerun. Set negative to disable.")

    # Misc output
    p.add_argument("--bundle-start-index", type=int, default=None, help="Optional inclusive bundle start index for debugging.")
    p.add_argument("--bundle-end-index", type=int, default=None, help="Optional inclusive bundle end index for debugging.")
    p.add_argument("--max-bundles", type=int, default=None, help="Optional hard cap on the number of bundles to scan.")
    p.add_argument("--save-detection-overlays", action="store_true")
    p.add_argument("--save-final-overlays", action="store_true")
    return p


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def maybe_load_transform(path: Optional[Path], name: str) -> Optional[np.ndarray]:
    if path is None:
        return None
    return load_world_from_camera(path, camera_name=name)


def print_and_save_filter_diagnostics(output_dir: Path, scan_records: List[Dict[str, Any]], pose_logs: Optional[List[Dict[str, Any]]] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(r.get("status", "unknown") for r in scan_records)
    print("\n=== scan status counts ===")
    for status, count in status_counts.most_common():
        print(f"{status}: {count}")

    # Save full records so the exact failed bundle and exception can be inspected.
    save_json(output_dir / "scan_records_debug.json", {"status_counts": dict(status_counts), "records": scan_records})
    if pose_logs is not None:
        pose_counts = Counter(r.get("status", "unknown") for r in pose_logs)
        print("\n=== pose filter status counts ===")
        for status, count in pose_counts.most_common():
            print(f"{status}: {count}")
        save_json(output_dir / "pose_filter_logs_debug.json", {"status_counts": dict(pose_counts), "records": pose_logs})

    print(f"\nSaved debug scan records: {output_dir / 'scan_records_debug.json'}")
    if pose_logs is not None:
        print(f"Saved debug pose logs   : {output_dir / 'pose_filter_logs_debug.json'}")

    # Print short examples for common failures.
    examples: Dict[str, List[Dict[str, Any]]] = {}
    for r in scan_records:
        status = r.get("status", "unknown")
        if status == "keep_pre_pose_filter":
            continue
        examples.setdefault(status, [])
        if len(examples[status]) < 5:
            examples[status].append(r)
    if examples:
        print("\n=== failure examples ===")
        for status, rows in examples.items():
            print(f"\n[{status}]")
            for row in rows:
                compact = {k: row.get(k) for k in row.keys() if k in [
                    "bundle_name", "bundle_index", "robot_abs_delay_ms", "robot_delay_ms", "match_span_ms",
                    "camera_host_delta_ms", "camera_match_delta_ms", "reproj_rms", "num_markers", "num_charuco", "error"
                ]}
                print(json.dumps(compact, ensure_ascii=False))

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.camera_name:
        parser.error("--camera-name is missing or empty: pass --camera-name oak_<MXID> (the bundle sub-directory of the camera "
                     "that saw the board), or omit the flag and set AURA_HANDEYE_OAK_ID=oak_<MXID> in config/aura.env and source it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    T_world_from_cam = load_world_from_camera(args.world_from_camera_file, camera_name=args.camera_name)

    if args.intrinsic_file is not None:
        K, dist, image_size = load_intrinsics(args.intrinsic_file)
    else:
        K, dist, image_size = load_intrinsics_from_session(args.session_root, args.camera_name)

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dictionary])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length,
        args.marker_length,
        aruco_dict,
    )
    detector = cv2.aruco.CharucoDetector(board)

    prefiltered_samples, scan_records = load_candidate_samples(
        session_root=args.session_root,
        camera_name=args.camera_name,
        T_world_from_cam=T_world_from_cam,
        K=K,
        dist=dist,
        image_size=image_size,
        board=board,
        detector=detector,
        args=args,
    )

    pose_selected, pose_logs = apply_pose_diversity_filter(
        prefiltered_samples,
        min_pos_m=args.min_position_diff_m,
        min_rot_deg=args.min_rotation_diff_deg,
        max_pos_step_m=args.max_position_step_m,
        max_rot_step_deg=args.max_rotation_step_deg,
    )

    if args.diagnose_only:
        print_and_save_filter_diagnostics(args.output_dir, scan_records, pose_logs)
        print(f"\nDiagnose only: prefilter kept {len(prefiltered_samples)}, pose diversity kept {len(pose_selected)}.")
        return

    if len(pose_selected) < 4:
        print_and_save_filter_diagnostics(args.output_dir, scan_records, pose_logs)
        raise RuntimeError(
            f"Only {len(pose_selected)} samples remain after filtering; need at least 4. "
            f"See {args.output_dir / 'scan_records_debug.json'} for exact drop reasons. "
            f"Try relaxing delay thresholds, reprojection RMS or pose diversity thresholds; for 'Image size mismatch' drops, pass intrinsics fitted at the recording resolution (--ignore-image-size-check is only for an image_size stored as (H,W))."
        )

    init_X = maybe_load_transform(args.init_ee_board_file, name="T_ee_from_board")
    if init_X is None:
        init_X = estimate_T_ee_from_board_via_relative_motion(
            pose_selected,
            rot_weight=args.rot_weight,
            trans_weight=args.trans_weight,
            max_pairs=args.motion_max_pairs,
            init_T=None,
        )

    init_A = maybe_load_transform(args.init_world_base_file, name="T_world_from_base")
    if init_A is None:
        init_A = estimate_T_world_from_base_given_X(pose_selected, init_X)

    T_world_from_base, T_ee_from_board, result = joint_refine(
        pose_selected,
        T_world_from_base_init=init_A,
        T_ee_from_board_init=init_X,
        rot_weight=args.rot_weight,
        trans_weight=args.trans_weight,
    )

    errors_first = compute_per_sample_errors(pose_selected, T_world_from_base, T_ee_from_board)

    final_max_trans = None if args.final_max_trans_err_m is not None and args.final_max_trans_err_m < 0 else args.final_max_trans_err_m
    final_max_rot = None if args.final_max_rot_err_deg is not None and args.final_max_rot_err_deg < 0 else args.final_max_rot_err_deg
    outlier_logs: List[Dict[str, Any]] = []
    final_samples = list(pose_selected)
    final_errors = errors_first

    if final_max_trans is not None or final_max_rot is not None:
        filtered_again, outlier_logs = reject_outliers(pose_selected, errors_first, final_max_trans, final_max_rot)
        if len(filtered_again) >= 4 and len(filtered_again) < len(pose_selected):
            T_world_from_base, T_ee_from_board, result = joint_refine(
                filtered_again,
                T_world_from_base_init=T_world_from_base,
                T_ee_from_board_init=T_ee_from_board,
                rot_weight=args.rot_weight,
                trans_weight=args.trans_weight,
            )
            final_samples = filtered_again
            final_errors = compute_per_sample_errors(final_samples, T_world_from_base, T_ee_from_board)

    if args.save_final_overlays:
        axis_length = args.axis_length if args.axis_length is not None else (args.square_length * 2.0)
        out_dir = args.output_dir / "detection_overlays_final"
        out_dir.mkdir(parents=True, exist_ok=True)
        for s, e in zip(final_samples, final_errors):
            img = cv2.imread(str(s.image_path))
            footer = f"delay={s.robot_delay_ms:.1f}ms terr={e['residual_translation_m']*1000:.1f}mm rerr={e['residual_rotation_deg']:.2f}deg"
            overlay = draw_overlay(img, K, dist, s.det, axis_length, footer=footer)
            cv2.imwrite(str(out_dir / f"{s.bundle_name}_{args.camera_name}.png"), overlay)

    summary = {
        "camera_name": args.camera_name,
        "session_root": str(args.session_root),
        "world_from_camera_file": str(args.world_from_camera_file),
        "image_size": list(image_size),
        "filters": {
            "max_robot_delay_ms": args.max_robot_delay_ms,
            "max_match_span_ms": args.max_match_span_ms,
            "max_camera_host_delta_ms": args.max_camera_host_delta_ms,
            "max_camera_match_delta_ms": args.max_camera_match_delta_ms,
            "min_markers": args.min_markers,
            "min_charuco": args.min_charuco,
            "max_reproj_rms": args.max_reproj_rms,
            "min_position_diff_m": args.min_position_diff_m,
            "min_rotation_diff_deg": args.min_rotation_diff_deg,
            "max_position_step_m": args.max_position_step_m,
            "max_rotation_step_deg": args.max_rotation_step_deg,
            "final_max_trans_err_m": final_max_trans,
            "final_max_rot_err_deg": final_max_rot,
            "bundle_start_index": args.bundle_start_index,
            "bundle_end_index": args.bundle_end_index,
            "max_bundles": args.max_bundles,
            "prefer_joint_fk": args.prefer_joint_fk,
            "extra_ee_z_m": args.extra_ee_z_m,
        },
        "counts": {
            "bundle_dirs": len([p for p in args.session_root.iterdir() if p.is_dir() and p.name.startswith('bundle_')]),
            "pass_detection_and_delay": len(prefiltered_samples),
            "pass_pose_diversity": len(pose_selected),
            "final_inliers": len(final_samples),
        },
        "optimization": {
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "status": int(result.status),
            "message": str(result.message),
        },
        "T_world_from_camera": T_world_from_cam.tolist(),
        "T_world_from_base": T_world_from_base.tolist(),
        "T_base_from_world": invert_T(T_world_from_base).tolist(),
        "T_ee_from_board": T_ee_from_board.tolist(),
        "T_board_from_ee": invert_T(T_ee_from_board).tolist(),
        "error_summary": {
            "mean_translation_mm": float(1000.0 * np.mean([e["residual_translation_m"] for e in final_errors])) if final_errors else None,
            "median_translation_mm": float(1000.0 * np.median([e["residual_translation_m"] for e in final_errors])) if final_errors else None,
            "max_translation_mm": float(1000.0 * np.max([e["residual_translation_m"] for e in final_errors])) if final_errors else None,
            "mean_rotation_deg": float(np.mean([e["residual_rotation_deg"] for e in final_errors])) if final_errors else None,
            "median_rotation_deg": float(np.median([e["residual_rotation_deg"] for e in final_errors])) if final_errors else None,
            "max_rotation_deg": float(np.max([e["residual_rotation_deg"] for e in final_errors])) if final_errors else None,
        },
        "per_sample_errors": final_errors,
        "scan_records": scan_records,
        "pose_filter_logs": pose_logs,
        "outlier_logs": outlier_logs,
    }

    save_json(args.output_dir / "robot_calib_result.json", summary)
    np.savez(
        args.output_dir / "robot_calib_result.npz",
        T_world_from_camera=T_world_from_cam,
        T_world_from_base=T_world_from_base,
        T_base_from_world=invert_T(T_world_from_base),
        T_ee_from_board=T_ee_from_board,
        T_board_from_ee=invert_T(T_ee_from_board),
    )

    print("=" * 80)
    print(f"Camera                    : {args.camera_name}")
    print(f"Pass detection+delay      : {len(prefiltered_samples)}")
    print(f"Pass pose diversity       : {len(pose_selected)}")
    print(f"Final inliers             : {len(final_samples)}")
    if final_errors:
        terr = [e["residual_translation_m"] for e in final_errors]
        rerr = [e["residual_rotation_deg"] for e in final_errors]
        print(f"Residual translation (mm) : mean={1000*np.mean(terr):.2f} median={1000*np.median(terr):.2f} max={1000*np.max(terr):.2f}")
        print(f"Residual rotation (deg)   : mean={np.mean(rerr):.3f} median={np.median(rerr):.3f} max={np.max(rerr):.3f}")
    print("T_world_from_base =")
    print(T_world_from_base)
    print("T_ee_from_board =")
    print(T_ee_from_board)
    print(f"Saved JSON: {args.output_dir / 'robot_calib_result.json'}")
    print(f"Saved NPZ : {args.output_dir / 'robot_calib_result.npz'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
