#!/usr/bin/env python3
r"""
Execute a selected AnyGrasp pose on a Franka/FR3 robot.

This script is intentionally staged:
  1) load AnyGrasp pose + calibration transform
  2) convert pose into Franka base frame
  3) print and sanity-check the target
  4) check Cartesian workspace for target/hover/final/lift
  5) check Joint 4 at runtime before/after motions
  6) optionally connect to robot and move: hover -> descend -> close -> lift

execute_anygrasp_franka_record_lift.py imports this file as a sibling module (keep
both in the same directory) and builds its command line on build_parser().

Supported grasp JSON formats:
  A) anygrasp_results/grasps.json or best_grasp.json exported by
     aura/grasping/anygrasp_poses_from_pointcloud.py, or grasps_table_ik_safe.json
     (the pipeline input) from aura/grasping/filter_grasps_table_collision_pyroki.py
     fields: translation, rotation_matrix, width, score, rank
  B) legacy selected-pose JSON (not produced by this repository)
     fields: selected_grasp.base_pose_matrix_4x4, etc.

IMPORTANT:
  - The debug z-flip used in visualization is NOT applied here.
  - By default this is a dry-run. The robot moves only with --execute.
  - For the first real run, use --hover-only first.
  - --rank indexes the JSON list (for grasps_table_ik_safe.json: the safe-list
    index, not the AnyGrasp rank printed as rank=).
  - --motion-speed / --descend-speed are fr3 move_to_pose speed fractions (0-1)
    here, but m/s in execute_anygrasp_franka_record_lift.py.

Run from the repository root with config/aura.env loaded
(set -a; source config/aura.env; set +a). --ip defaults to $AURA_ROBOT_IP;
--execute needs the robot uv project ($AURA_ROBOT_PROJECT, provides fr3).
--rot-offset-rpy 0 1.57079632679 0 is the AnyGrasp-to-TCP convention used by the
pipeline and by the PyRoki filter.

Example dry run:
  uv run --project "$AURA_ROBOT_PROJECT" python aura/robot/execute_anygrasp_franka.py \
      --calib "$AURA_CALIB_ROOT/robot_calibration_results/robot_calib_result.npz" \
      --grasp-json "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp_results/grasps_table_ik_safe.json" \
      --rank 0 \
      --pose-frame world \
      --rot-offset-rpy 0 1.57079632679 0

First real safety test, hover only:
  uv run --project "$AURA_ROBOT_PROJECT" python aura/robot/execute_anygrasp_franka.py \
      --calib "$AURA_CALIB_ROOT/robot_calibration_results/robot_calib_result.npz" \
      --grasp-json "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp_results/grasps_table_ik_safe.json" \
      --rank 0 \
      --pose-frame world \
      --rot-offset-rpy 0 1.57079632679 0 \
      --execute --hover-only --home

Full staged grasp:
  uv run --project "$AURA_ROBOT_PROJECT" python aura/robot/execute_anygrasp_franka.py \
      --calib "$AURA_CALIB_ROOT/robot_calibration_results/robot_calib_result.npz" \
      --grasp-json "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp_results/grasps_table_ik_safe.json" \
      --rank 0 \
      --pose-frame world \
      --rot-offset-rpy 0 1.57079632679 0 \
      --execute --home

  The run ends with the object held at the lift pose: this script neither releases it
  nor homes (the lift one-shot does both in its stage 02b). Release it yourself, for
  example from Franka Desk, before the next run: a rerun opens the gripper right after
  connecting (after homing, with --home) without a separate prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


# Default Cartesian workspace box in the FR3 base frame (metres).
WORKSPACE_DEFAULT = {
    "x": (0.20, 0.80),
    "y": (-0.40, 0.40),
    "z": (0.00, 0.60),
}


def as_matrix4(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {arr.shape}")
    return arr


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {R.shape}")
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_calib(calib_path: Path) -> Dict[str, np.ndarray]:
    """Load calibration from .npz or .json."""
    if not calib_path.exists():
        raise FileNotFoundError(f"calibration file not found: {calib_path}")

    if calib_path.suffix.lower() == ".npz":
        data = np.load(calib_path)
        out = {k: np.asarray(data[k], dtype=float) for k in data.files}
    elif calib_path.suffix.lower() == ".json":
        with open(calib_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out = {}
        for key in [
            "T_world_from_camera",
            "T_world_from_base",
            "T_base_from_world",
            "T_ee_from_board",
            "T_board_from_ee",
        ]:
            if key in raw:
                out[key] = np.asarray(raw[key], dtype=float)
    else:
        raise ValueError("calib must be .npz or .json")

    if "T_base_from_world" not in out:
        if "T_world_from_base" not in out:
            raise KeyError("calib needs T_base_from_world or T_world_from_base")
        out["T_base_from_world"] = np.linalg.inv(out["T_world_from_base"])

    if "T_world_from_camera" in out:
        out["T_base_from_camera"] = out["T_base_from_world"] @ out["T_world_from_camera"]

    return out


def select_record(obj: Any, rank: int) -> Dict[str, Any]:
    """Return one grasp record from several possible JSON layouts."""
    if isinstance(obj, list):
        if rank < 0 or rank >= len(obj):
            raise IndexError(f"rank {rank} outside grasps list length {len(obj)}")
        return obj[rank]

    if not isinstance(obj, dict):
        raise TypeError("grasp JSON must be a dict or list")

    if "selected_grasp" in obj:
        return obj["selected_grasp"]

    # best_grasp.json is already a single record.
    if any(k in obj for k in ["pose_matrix_4x4", "rotation_matrix", "base_pose_matrix_4x4"]):
        return obj

    if "grasps" in obj and isinstance(obj["grasps"], list):
        return obj["grasps"][rank]

    raise KeyError("could not find grasp record in JSON")


def load_grasp_pose(grasp_json: Path, rank: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load selected grasp as T_input_from_grasp plus metadata."""
    with open(grasp_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rec = select_record(obj, rank)

    # Already transformed base-frame selected pose.
    if "base_pose_matrix_4x4" in rec:
        T = as_matrix4(rec["base_pose_matrix_4x4"], "base_pose_matrix_4x4")
        rec = dict(rec)
        rec.setdefault("_detected_frame", "base")
        return T, rec

    # Regular AnyGrasp exported pose.
    if "pose_matrix_4x4" in rec:
        T = as_matrix4(rec["pose_matrix_4x4"], "pose_matrix_4x4")
        return T, rec

    if "rotation_matrix" in rec and "translation" in rec:
        T = make_T(np.asarray(rec["rotation_matrix"], dtype=float), np.asarray(rec["translation"], dtype=float))
        return T, rec

    # Fallback for selected pose split into base fields.
    if "base_rotation_matrix" in rec and "base_translation_m" in rec:
        T = make_T(np.asarray(rec["base_rotation_matrix"], dtype=float), np.asarray(rec["base_translation_m"], dtype=float))
        rec = dict(rec)
        rec.setdefault("_detected_frame", "base")
        return T, rec

    if "world_translation_m" in rec and "world_rotation_matrix" in rec:
        T = make_T(np.asarray(rec["world_rotation_matrix"], dtype=float), np.asarray(rec["world_translation_m"], dtype=float))
        rec = dict(rec)
        rec.setdefault("_detected_frame", "world")
        return T, rec

    raise KeyError("grasp record does not contain pose_matrix_4x4 or rotation_matrix+translation")


def transform_pose_to_base(T_input_from_grasp: np.ndarray, pose_frame: str, calib: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert AnyGrasp pose from input frame to Franka base frame."""
    pose_frame = pose_frame.lower()
    if pose_frame == "base":
        return T_input_from_grasp.copy()
    if pose_frame == "world":
        return calib["T_base_from_world"] @ T_input_from_grasp
    if pose_frame in ("camera", "cam"):
        if "T_base_from_camera" not in calib:
            raise KeyError("pose_frame=camera requires T_world_from_camera in calibration")
        return calib["T_base_from_camera"] @ T_input_from_grasp
    raise ValueError("pose_frame must be one of: world, camera, base")


def rpy_from_matrix(R: np.ndarray) -> Tuple[float, float, float]:
    """Return roll, pitch, yaw for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    R = np.asarray(R, dtype=float)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def apply_offsets(T_base_from_grasp: np.ndarray, tcp_offset_grasp_frame, rot_offset_rpy) -> np.ndarray:
    """Post-multiply the rotation offset, then add the TCP offset in the rotated (target) frame."""
    T = T_base_from_grasp.copy()

    if rot_offset_rpy is not None:
        rr, pp, yy = rot_offset_rpy
        T[:3, :3] = T[:3, :3] @ rpy_to_matrix(rr, pp, yy)

    if tcp_offset_grasp_frame is not None:
        off = np.asarray(tcp_offset_grasp_frame, dtype=float).reshape(3)
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ off

    return T


def check_rotation(R: np.ndarray) -> Tuple[bool, str]:
    det = float(np.linalg.det(R))
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    ok = abs(det - 1.0) < 1e-2 and ortho_err < 1e-2
    return ok, f"det(R)={det:.5f}, orthonormal_error={ortho_err:.3e}"


def check_workspace(xyz: np.ndarray, workspace: Dict[str, Tuple[float, float]]) -> Tuple[bool, str]:
    for i, axis in enumerate("xyz"):
        lo, hi = workspace[axis]
        v = float(xyz[i])
        if not (lo <= v <= hi):
            return False, f"{axis}={v:+.3f} outside [{lo:+.2f}, {hi:+.2f}]"
    return True, "inside workspace"


def ask_continue(message: str, assume_yes: bool = False) -> None:
    if assume_yes:
        print(f"[auto-confirm] {message}")
        return
    ans = input(f"{message} Type 'yes' to continue: ").strip().lower()
    if ans != "yes":
        raise KeyboardInterrupt("aborted by user")


def get_record_width(rec: Dict[str, Any]) -> float | None:
    for k in ["width", "anygrasp_width_m"]:
        if k in rec:
            try:
                return float(rec[k])
            except Exception:
                pass
    return None


def try_get_joint_angles(robot) -> np.ndarray | None:
    """Best-effort joint angle reader for different fr3 wrapper versions.

    Returns q as shape (7,) in radians, or None if this fr3 wrapper does not
    expose a joint-state reader.
    """
    for name in ["get_joint_angles", "get_joints", "get_joint_positions", "get_q"]:
        fn = getattr(robot, name, None)
        if callable(fn):
            try:
                q = np.asarray(fn(), dtype=float).reshape(-1)
                if q.size >= 7:
                    return q[:7]
            except Exception:
                pass

    fn = getattr(robot, "get_state", None)
    if callable(fn):
        try:
            state = fn()
            if isinstance(state, dict):
                for key in ["q", "joint_angles", "joints", "joint_positions"]:
                    if key in state:
                        q = np.asarray(state[key], dtype=float).reshape(-1)
                        if q.size >= 7:
                            return q[:7]
            for key in ["q", "joint_angles", "joints", "joint_positions"]:
                if hasattr(state, key):
                    q = np.asarray(getattr(state, key), dtype=float).reshape(-1)
                    if q.size >= 7:
                        return q[:7]
        except Exception:
            pass

    return None


def check_joint4_soft_limit(robot, args, stage: str) -> None:
    """Abort if current Joint 4 is outside a software-defined safe range."""
    if args.disable_joint4_check:
        return

    q = try_get_joint_angles(robot)
    if q is None:
        print(f"[WARN] Could not read joint angles at stage={stage!r}; skipping Joint-4 check.")
        return

    j4 = float(q[3])
    print(
        f"[safety] {stage}: Joint 4 = {j4:+.4f} rad "
        f"({math.degrees(j4):+.1f} deg), "
        f"soft range=[{args.joint4_soft_min:+.3f}, {args.joint4_soft_max:+.3f}] rad"
    )

    if j4 < args.joint4_soft_min or j4 > args.joint4_soft_max:
        raise RuntimeError(
            f"Unsafe current Joint 4 at {stage}: q4={j4:+.4f} rad "
            f"({math.degrees(j4):+.1f} deg), outside software range "
            f"[{args.joint4_soft_min:+.3f}, {args.joint4_soft_max:+.3f}] rad. "
            "Move the robot away from the joint limit in Franka Desk/guiding mode before retrying."
        )


def build_workspace_from_args(args) -> Dict[str, Tuple[float, float]]:
    """Build Cartesian workspace limits from CLI arguments."""
    return {
        "x": (float(args.workspace_x_min), float(args.workspace_x_max)),
        "y": (float(args.workspace_y_min), float(args.workspace_y_max)),
        "z": (float(args.workspace_z_min), float(args.workspace_z_max)),
    }


def check_planned_cartesian_workspace(
    x: float,
    y: float,
    hover_z: float,
    final_z: float,
    lift_z: float,
    args,
) -> None:
    """Check hover/final/lift poses against Cartesian workspace.

    Applies the same x/y/z box as the raw-target check in print_pose_summary()
    to every planned z stage.
    It does not impose an extra "table height" threshold.
    """
    if args.disable_workspace_check:
        print("[safety] Cartesian workspace check disabled.")
        return

    workspace = build_workspace_from_args(args)
    stages = {
        "hover": np.array([x, y, hover_z], dtype=float),
        "final": np.array([x, y, final_z], dtype=float),
        "lift": np.array([x, y, lift_z], dtype=float),
    }

    print("\n========== CARTESIAN WORKSPACE GUARD ==========")
    print(
        f"x={workspace['x']}  y={workspace['y']}  z={workspace['z']} "
        "(checked for hover/final/lift)"
    )

    bad = []
    for name, xyz in stages.items():
        ok, why = check_workspace(xyz, workspace)
        print(
            f"[safety] {name:5s}: "
            f"x={xyz[0]:+.4f}, y={xyz[1]:+.4f}, z={xyz[2]:+.4f} -> "
            f"{'OK' if ok else 'BAD'} {why if not ok else ''}"
        )
        if not ok:
            bad.append((name, why))

    if bad and not args.allow_outside_workspace:
        joined = "; ".join([f"{name}: {why}" for name, why in bad])
        raise RuntimeError(
            "Cartesian workspace guard rejected this grasp: "
            f"{joined}. Use another rank, adjust offsets/workspace limits, "
            "or pass --allow-outside-workspace only if you are sure."
        )


def print_pose_summary(T: np.ndarray, rec: Dict[str, Any], args) -> Tuple[float, float, float]:
    xyz = T[:3, 3]
    rpy = rpy_from_matrix(T[:3, :3])
    width = get_record_width(rec)

    print("\n========== SELECTED ANYGRASP TARGET ==========")
    print(f"rank/score: rank={rec.get('rank', args.rank)}  score={rec.get('score', 'n/a')}")
    if width is not None:
        print(f"AnyGrasp required width: {width*1000.0:.1f} mm")
    print("T_base_from_target =")
    np.set_printoptions(precision=6, suppress=True)
    print(T)
    print(f"base xyz:  x={xyz[0]:+.4f}  y={xyz[1]:+.4f}  z={xyz[2]:+.4f} m")
    print(f"base rpy:  roll={rpy[0]:+.4f}  pitch={rpy[1]:+.4f}  yaw={rpy[2]:+.4f} rad")
    print(f"base rpy:  roll={math.degrees(rpy[0]):+.2f}  pitch={math.degrees(rpy[1]):+.2f}  yaw={math.degrees(rpy[2]):+.2f} deg")

    okR, msgR = check_rotation(T[:3, :3])
    workspace = build_workspace_from_args(args)
    okW, msgW = check_workspace(xyz, workspace)
    print(f"rotation check: {'OK' if okR else 'BAD'}  {msgR}")
    print(f"target workspace check: {'OK' if okW else 'BAD'}  {msgW}")

    if width is not None:
        if width > args.max_gripper_width and not args.allow_wide_grasp:
            raise RuntimeError(
                f"AnyGrasp width {width:.3f} m is wider than max_gripper_width "
                f"{args.max_gripper_width:.3f} m. Use a different rank or --allow-wide-grasp."
            )

    if not okR:
        raise RuntimeError("rotation matrix failed sanity check")
    if not okW and not args.allow_outside_workspace:
        raise RuntimeError("target outside workspace; use --allow-outside-workspace only if you are sure")

    return rpy


def execute_robot(T: np.ndarray, rpy: Tuple[float, float, float], args, rec: Dict[str, Any]) -> None:
    import fr3  # Imported only when actually executing on the robot machine.

    x, y, z = [float(v) for v in T[:3, 3]]
    roll, pitch, yaw = rpy

    hover_z = z + args.hover_z_offset
    final_z = z + args.final_z_offset
    lift_z = z + args.lift_z_offset

    print("\n========== ROBOT EXECUTION PLAN ==========")
    print(f"hover: x={x:+.4f}, y={y:+.4f}, z={hover_z:+.4f}")
    print(f"final: x={x:+.4f}, y={y:+.4f}, z={final_z:+.4f}")
    print(f"lift : x={x:+.4f}, y={y:+.4f}, z={lift_z:+.4f}")
    print(f"rpy  : roll={roll:+.4f}, pitch={pitch:+.4f}, yaw={yaw:+.4f}")

    check_planned_cartesian_workspace(x, y, hover_z, final_z, lift_z, args)

    if not args.yes:
        print("\nSafety reminder: run --hover-only before full grasp if this is a new calibration/object.")
    ask_continue("Connect to robot and start?", args.yes)

    with fr3.FR3(ip=args.ip) as robot:
        check_joint4_soft_limit(robot, args, stage="after_connect_before_home")

        if args.home:
            ask_continue("Home the robot?", args.yes)
            print("[robot] homing...")
            robot.home(blocking=True)
            check_joint4_soft_limit(robot, args, stage="after_home")

        if args.skip_open_gripper:
            print("[robot] skipping gripper open before hover.")
        else:
            print("[robot] opening gripper...")
            robot.open_gripper()

        check_joint4_soft_limit(robot, args, stage="before_hover")

        ask_continue("Move to HOVER pose?", args.yes)
        print(f"[motion] HOVER -> ({x:+.4f}, {y:+.4f}, {hover_z:+.4f})")
        robot.move_to_pose(x, y, hover_z, roll, pitch, yaw, speed=args.motion_speed, blocking=True)
        check_joint4_soft_limit(robot, args, stage="after_hover")

        if args.hover_only:
            print("[done] hover-only mode. No descent/grasp/lift executed.")
            return

        ask_continue("Descend to FINAL grasp pose?", args.yes)
        print(f"[motion] DESCEND -> ({x:+.4f}, {y:+.4f}, {final_z:+.4f})")
        if not args.skip_open_gripper:
            robot.open_gripper()
        check_joint4_soft_limit(robot, args, stage="before_descend")
        robot.move_to_pose(x, y, final_z, roll, pitch, yaw, speed=args.descend_speed, blocking=True)
        check_joint4_soft_limit(robot, args, stage="after_descend")

        grasp_width = args.grasp_width
        if args.use_anygrasp_width:
            w = get_record_width(rec)
            if w is None:
                raise RuntimeError("--use-anygrasp-width was set but grasp record has no width")
            grasp_width = max(0.0, min(float(w), args.max_gripper_width))

        ask_continue("Close gripper and grasp?", args.yes)
        print(
            f"[grasp] width={grasp_width:.4f} m, force={args.grasp_force:.1f} N, "
            f"epsilon=±{args.grasp_epsilon:.3f} m"
        )
        got = robot._gripper.grasp(
            width=grasp_width,
            speed=args.grasp_close_speed,
            force=args.grasp_force,
            epsilon_inner=args.grasp_epsilon,
            epsilon_outer=args.grasp_epsilon,
        )
        gs = robot._gripper.read_state()
        print(
            f"[grasp] reported={'SUCCESS' if got else 'FAILED'}  "
            f"actual width={gs['width']*1000.0:.1f} mm  is_grasped={gs['is_grasped']}"
        )

        if not args.lift_even_if_not_grasped and not gs["is_grasped"]:
            print("[abort] gripper reports no object. Re-opening and skipping lift.")
            robot.open_gripper()
            return

        ask_continue("Lift object?", args.yes)
        print(f"[motion] LIFT -> ({x:+.4f}, {y:+.4f}, {lift_z:+.4f})")
        check_joint4_soft_limit(robot, args, stage="before_lift")
        robot.move_to_pose(x, y, lift_z, roll, pitch, yaw, speed=args.motion_speed, blocking=True)
        check_joint4_soft_limit(robot, args, stage="after_lift")
        gs2 = robot._gripper.read_state()
        print(f"[post-lift] width={gs2['width']*1000.0:.1f} mm  is_grasped={gs2['is_grasped']}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ip", default=os.environ.get("AURA_ROBOT_IP"), help="Franka/FR3 robot IP; defaults to $AURA_ROBOT_IP (config/aura.env)")
    ap.add_argument("--calib", required=True, help="robot_calib_result.npz or robot_calib_result.json")
    ap.add_argument("--grasp-json", required=True, help="grasps_table_ik_safe.json (pipeline input), AnyGrasp grasps.json, best_grasp.json, or a legacy selected-pose JSON")
    ap.add_argument("--rank", type=int, default=0, help="index into the --grasp-json list (for grasps_table_ik_safe.json: the safe-list index, the lift one-shot's --safe-index; not the AnyGrasp rank printed as rank=)")
    ap.add_argument(
        "--pose-frame",
        choices=["world", "camera", "base"],
        default="world",
        help="frame of AnyGrasp pose. For a legacy selected-pose JSON (base_pose_matrix_4x4) use base; for grasps on the fused observation point cloud (ChArUco world frame) use world.",
    )

    ap.add_argument("--execute", action="store_true", help="actually connect to robot and move")
    ap.add_argument("--hover-only", action="store_true", help="only move to hover pose, no descent/grasp/lift; execute_anygrasp_franka_record_lift.py hangs at hover until Ctrl-C, so do hover tests with execute_anygrasp_franka.py")
    ap.add_argument("--yes", action="store_true", help="auto-confirm all staged prompts")
    ap.add_argument("--home", action="store_true", help="home robot after connecting")

    ap.add_argument("--orientation-mode", choices=["anygrasp", "home"], default="anygrasp",
                    help="use AnyGrasp orientation, or use robot home/current orientation while keeping AnyGrasp XYZ; in execute_anygrasp_franka_record_lift.py use anygrasp: home needs --home-rpy within 0.5 deg of the filter's orientation")
    ap.add_argument("--home-rpy", nargs=3, type=float, default=None, metavar=("ROLL", "PITCH", "YAW"),
                    help="RPY to use when --orientation-mode home; if omitted, execute_anygrasp_franka.py reads the robot pose after connecting (--execute only) and execute_anygrasp_franka_record_lift.py stops with an error")

    ap.add_argument("--tcp-offset-grasp-frame", nargs=3, type=float, default=None, metavar=("DX", "DY", "DZ"),
                    help="TCP origin offset in meters, in the target frame after --rot-offset-rpy (pipeline rotation: z = AnyGrasp approach axis); the PyRoki filter ignores it, so execute_anygrasp_franka_record_lift.py allows at most 1 mm")
    ap.add_argument("--rot-offset-rpy", nargs=3, type=float, default=None, metavar=("DROLL", "DPITCH", "DYAW"),
                    help="optional rotation offset applied after AnyGrasp rotation, radians")

    ap.add_argument("--hover-z-offset", type=float, default=0.10, help="vertical hover offset above final target")
    ap.add_argument("--final-z-offset", type=float, default=0.00, help="vertical offset added to final grasp z")
    ap.add_argument("--lift-z-offset", type=float, default=0.15, help="vertical lift offset above final target")
    ap.add_argument("--motion-speed", type=float, default=0.10, help="hover/lift speed: fr3 move_to_pose speed fraction (0-1) in execute_anygrasp_franka.py; hover speed in m/s in execute_anygrasp_franka_record_lift.py (lift: --lift-speed)")
    ap.add_argument("--descend-speed", type=float, default=0.05, help="descent speed: fr3 move_to_pose speed fraction (0-1) in execute_anygrasp_franka.py; m/s in execute_anygrasp_franka_record_lift.py")

    ap.add_argument("--max-gripper-width", type=float, default=0.08, help="maximum allowed gripper opening, meters")
    ap.add_argument("--allow-wide-grasp", action="store_true", help="do not reject if AnyGrasp width exceeds max_gripper_width")
    ap.add_argument("--allow-outside-workspace", action="store_true", help="do not reject if pose is outside configured workspace (no effect in execute_anygrasp_franka_record_lift.py)")
    ap.add_argument("--disable-workspace-check", action="store_true",
                    help="skip the hover/final/lift workspace guard in execute_anygrasp_franka.py; the target check still rejects unless --allow-outside-workspace is passed; no effect in execute_anygrasp_franka_record_lift.py")

    # Cartesian workspace box (FR3 base frame, metres), checked for the target
    # and the hover/final/lift poses (not enforced by execute_anygrasp_franka_record_lift.py).
    ap.add_argument("--workspace-x-min", type=float, default=WORKSPACE_DEFAULT["x"][0])
    ap.add_argument("--workspace-x-max", type=float, default=WORKSPACE_DEFAULT["x"][1])
    ap.add_argument("--workspace-y-min", type=float, default=WORKSPACE_DEFAULT["y"][0])
    ap.add_argument("--workspace-y-max", type=float, default=WORKSPACE_DEFAULT["y"][1])
    ap.add_argument("--workspace-z-min", type=float, default=WORKSPACE_DEFAULT["z"][0])
    ap.add_argument("--workspace-z-max", type=float, default=WORKSPACE_DEFAULT["z"][1])

    ap.add_argument("--grasp-width", type=float, default=0.0,
                    help="target gripper width. 0.0 means close until contact")
    ap.add_argument("--use-anygrasp-width", action="store_true",
                    help="use AnyGrasp predicted width as gripper target width instead of --grasp-width")
    ap.add_argument("--grasp-force", type=float, default=35.0, help="grasp force in N")
    ap.add_argument("--grasp-epsilon", type=float, default=0.08, help="inner/outer grasp epsilon in meters")
    ap.add_argument("--grasp-close-speed", type=float, default=0.05, help="gripper close speed")
    ap.add_argument("--lift-even-if-not-grasped", action="store_true", help="lift even if gripper state says no object")

    # Preventive Joint-4 software guard. This does not replace Franka Desk recovery if
    # the robot is already in a reflex/joint-limit error state.
    ap.add_argument("--joint4-soft-min", type=float, default=-2.75,
                    help="software lower limit for Joint 4 in radians; abort if current q4 is lower (no effect in execute_anygrasp_franka_record_lift.py)")
    ap.add_argument("--joint4-soft-max", type=float, default=+2.75,
                    help="software upper limit for Joint 4 in radians; abort if current q4 is higher (no effect in execute_anygrasp_franka_record_lift.py)")
    ap.add_argument("--disable-joint4-check", action="store_true",
                    help="disable runtime Joint-4 software check (no effect in execute_anygrasp_franka_record_lift.py)")
    ap.add_argument("--skip-open-gripper", action="store_true",
                    help="skip robot.open_gripper() before hover/descent")
    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.execute and not args.ip:
        parser.error("--execute needs the robot IP: set AURA_ROBOT_IP in config/aura.env and load it (set -a; source config/aura.env; set +a) or pass --ip <ROBOT_IP>")

    calib = load_calib(Path(args.calib))
    T_input_from_grasp, rec = load_grasp_pose(Path(args.grasp_json), args.rank)

    # Auto-detect base frame from selected_anygrasp_pose.json unless user explicitly chose otherwise.
    pose_frame = rec.get("_detected_frame", args.pose_frame)
    if rec.get("_detected_frame") == "base" and args.pose_frame != "base":
        print("[info] JSON contains base_pose_matrix_4x4; overriding pose_frame to base.")
        pose_frame = "base"

    T_base_from_grasp = transform_pose_to_base(T_input_from_grasp, pose_frame, calib)
    T_base_from_target = apply_offsets(T_base_from_grasp, args.tcp_offset_grasp_frame, args.rot_offset_rpy)

    # Optionally replace orientation with user-provided home RPY. This is safer for first hover tests.
    if args.orientation_mode == "home":
        if args.home_rpy is not None:
            T_base_from_target[:3, :3] = rpy_to_matrix(*args.home_rpy)
        elif not args.execute:
            raise RuntimeError("--orientation-mode home without --execute needs --home-rpy R P Y for dry-run")
        else:
            # Will be read after connecting below; for planning print keep AnyGrasp orientation for now.
            print("[info] --orientation-mode home: will read current robot RPY after connecting.")

    rpy = print_pose_summary(T_base_from_target, rec, args)

    # Dry-run also checks the full hover/final/lift Cartesian workspace plan.
    x0, y0, z0 = [float(v) for v in T_base_from_target[:3, 3]]
    check_planned_cartesian_workspace(
        x0, y0,
        z0 + args.hover_z_offset,
        z0 + args.final_z_offset,
        z0 + args.lift_z_offset,
        args,
    )

    if not args.execute:
        print("\n[dry-run] Robot was NOT connected. Add --execute to move the robot.")
        print("Recommended first real command: add --execute --hover-only --home")
        return

    if args.orientation_mode == "home" and args.home_rpy is None:
        # Need to connect once to read pose before executing staged sequence.
        import fr3
        ask_continue("Connect to robot to read current/home orientation?", args.yes)
        with fr3.FR3(ip=args.ip) as robot:
            if args.home:
                print("[robot] homing before reading orientation...")
                robot.home(blocking=True)
            pose = robot.get_pose()
            args.home_rpy = (pose.roll, pose.pitch, pose.yaw)
            print(f"[robot] using current/home RPY: {args.home_rpy}")
        T_base_from_target[:3, :3] = rpy_to_matrix(*args.home_rpy)
        rpy = print_pose_summary(T_base_from_target, rec, args)
        x0, y0, z0 = [float(v) for v in T_base_from_target[:3, 3]]
        check_planned_cartesian_workspace(
            x0, y0,
            z0 + args.hover_z_offset,
            z0 + args.final_z_offset,
            z0 + args.lift_z_offset,
            args,
        )
        args.home = False  # avoid homing a second time unless you remove this line

    execute_robot(T_base_from_target, rpy, args, rec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print(f"\n[abort] {e}")
        sys.exit(130)
