#!/usr/bin/env python3
"""
Fit Franka gripper outer fingertip translations wrt end-effector frame from table-touch trajectories.

Usage (from the repository root):
  python aura/calibration/fit_fingertip_translation_from_json.py <contacts>.json -o <fit>.json

The result is printed to stdout; -o/--output also writes it (an existing file is overwritten).
The grasp filter and the lift one-shot read the fit from
  $AURA_CALIB_ROOT/robot_calibration_results/fingertip_calibration_results/franka_fingertip_table_calibration_simple.json
(see assets/README.md). The output has no "trajectories" block: never pass the input file
as -o, or the observations are lost. Write to a new file and copy or merge it into place.

Input JSON schema expected (example: assets/franka_fingertip_table_calibration_simple.json):
{
  "trajectories": [
    {
      "sample_id": "sample_01",
      "observed_contacts": ["right_outer", "left_outer"],
      "ee_position_in_base_m": [x, y, z],
      "ee_orientation_rpy_rad": [roll, pitch, yaw]
    }, ...
  ]
}
The pose is the end-effector (libfranka O_T_EE) pose in the robot base frame, e.g. as returned
by fr3.FR3().get_pose() (third_party/fr3). Contact names must contain "right" or "left"; when
observed_contacts is missing or empty, a contact_type containing "both", "right" or "left"
(checked in that order) is used.
At least 6 contact observations are required; vary the orientation between touches.

Model assumption:
  right_outer_tip_in_ee = [x, +y, z]
  left_outer_tip_in_ee  = [x, -y, z]
All observed contact points lie on one common table plane in robot base frame.

This script uses only numpy. It alternates between:
  1) fitting the table plane from current contact points by SVD/PCA;
  2) solving the fingertip translation q=[x,y,z] by linear least squares given the plane.

Output JSON: end_effector_fingertip_translation (right/left_outer_tip_in_ee_m and _mm,
outer_to_outer_distance_m/_mm), fitted_table_plane_in_base, residual_summary and
trajectory_residuals. Downstream code reads only right_outer_tip_in_ee_m and left_outer_tip_in_ee_m.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll-pitch-yaw to rotation matrix.

    Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), which is the common
    fixed-axis/base-frame RPY convention used by many robotics scripts.
    It is the same convention as euler_to_matrix() in third_party/fr3/src/fr3/_kinematics.py,
    so the roll/pitch/yaw returned by fr3.FR3().get_pose() can be used as-is.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )
    return rz @ ry @ rx


def contact_sign(contact_name: str) -> int:
    """right tip uses +y, left tip uses -y in EE frame."""
    name = contact_name.lower()
    if "right" in name:
        return +1
    if "left" in name:
        return -1
    raise ValueError(f"Cannot infer left/right from contact name: {contact_name!r}")


def load_observations(json_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(json_path.read_text())
    trajectories = data["trajectories"]

    observations: List[Dict[str, Any]] = []
    for traj in trajectories:
        sample_id = traj.get("sample_id", f"sample_{len(observations):03d}")
        p_ee_base = np.asarray(traj["ee_position_in_base_m"], dtype=float)
        rpy = np.asarray(traj["ee_orientation_rpy_rad"], dtype=float)
        R_ee_base = rpy_to_rotation_matrix(*rpy)

        contacts = traj.get("observed_contacts")
        if not contacts:
            # Fallback from contact_type, useful if your future JSON omits observed_contacts.
            ct = traj.get("contact_type", "").lower()
            if "both" in ct:
                contacts = ["right_outer", "left_outer"]
            elif "right" in ct:
                contacts = ["right_outer"]
            elif "left" in ct:
                contacts = ["left_outer"]
            else:
                raise ValueError(
                    f"Trajectory {sample_id} has neither observed_contacts nor clear contact_type."
                )

        for contact in contacts:
            observations.append(
                {
                    "sample_id": sample_id,
                    "contact": contact,
                    "sign": contact_sign(contact),
                    "p_ee_base": p_ee_base,
                    "R_ee_base": R_ee_base,
                }
            )

    if len(observations) < 6:
        raise ValueError(
            f"Only {len(observations)} contact observations found. More varied samples are recommended."
        )
    return observations


def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit plane n^T x + d = 0 from 3D points.
    Returns unit normal n and scalar d.
    """
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    # Keep the normal direction consistent with this dataset: table normal roughly -base_z.
    # Flipping normal does not change the plane; it only changes residual signs.
    if normal[2] > 0:
        normal = -normal

    d = -float(normal @ centroid)
    return normal, d


def contact_points_from_q(observations: List[Dict[str, Any]], q_xyz: np.ndarray) -> np.ndarray:
    """
    q_xyz = [x, y_abs, z] in EE frame.
    right tip: [x, +y_abs, z]
    left tip:  [x, -y_abs, z]
    """
    points = []
    for obs in observations:
        tip_ee = np.array([q_xyz[0], obs["sign"] * q_xyz[1], q_xyz[2]], dtype=float)
        p_contact_base = obs["p_ee_base"] + obs["R_ee_base"] @ tip_ee
        points.append(p_contact_base)
    return np.vstack(points)


def solve_q_given_plane(
    observations: List[Dict[str, Any]], normal: np.ndarray, d: float
) -> np.ndarray:
    """
    Given plane n,d, solve q=[x,y_abs,z] by linear least squares:
      n^T (p_ee + R * diag(1, sign, 1) * q) + d = 0
    """
    A_rows = []
    b_vals = []
    for obs in observations:
        S = np.diag([1.0, float(obs["sign"]), 1.0])
        A_rows.append(normal @ obs["R_ee_base"] @ S)
        b_vals.append(-(normal @ obs["p_ee_base"] + d))

    A = np.vstack(A_rows)
    b = np.asarray(b_vals, dtype=float)
    q, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Make sure the absolute y offset is positive.
    if q[1] < 0:
        q[1] = -q[1]
    return q


def residuals_for_q(
    observations: List[Dict[str, Any]], q_xyz: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    points = contact_points_from_q(observations, q_xyz)
    normal, d = fit_plane_svd(points)
    residuals = points @ normal + d
    return normal, d, residuals, points


def fit_fingertips(
    observations: List[Dict[str, Any]], max_iter: int = 100, tol: float = 1e-12
) -> Dict[str, Any]:
    """
    Multi-start alternating least squares to avoid poor initialization.
    """
    initial_guesses = []
    for x0 in [0.0, 0.001, -0.005, 0.005, 0.02, -0.02]:
        for y0 in [0.035, 0.04, 0.052, 0.06]:
            for z0 in [-0.02, 0.0, 0.005, 0.02, 0.05]:
                initial_guesses.append(np.array([x0, y0, z0], dtype=float))

    best = None
    for q in initial_guesses:
        previous_cost = np.inf
        for _ in range(max_iter):
            normal, d, residuals, _ = residuals_for_q(observations, q)
            q_new = solve_q_given_plane(observations, normal, d)
            normal_new, d_new, residuals_new, _ = residuals_for_q(observations, q_new)
            cost = float(np.mean(residuals_new**2))

            if np.linalg.norm(q_new - q) < tol or abs(previous_cost - cost) < tol**2:
                q = q_new
                break
            q = q_new
            previous_cost = cost

        normal, d, residuals, points = residuals_for_q(observations, q)
        rms = float(np.sqrt(np.mean(residuals**2)))
        max_abs = float(np.max(np.abs(residuals)))

        if best is None or rms < best["rms_residual_m"]:
            best = {
                "q_xyz_m": q,
                "plane_normal_base": normal,
                "plane_d_base_m": d,
                "residuals_m": residuals,
                "contact_points_base_m": points,
                "rms_residual_m": rms,
                "max_abs_residual_m": max_abs,
            }

    assert best is not None
    return best


def build_output(observations: List[Dict[str, Any]], fit: Dict[str, Any]) -> Dict[str, Any]:
    q = fit["q_xyz_m"]
    right = np.array([q[0], +q[1], q[2]], dtype=float)
    left = np.array([q[0], -q[1], q[2]], dtype=float)

    residual_items = []
    for obs, residual, point in zip(
        observations, fit["residuals_m"], fit["contact_points_base_m"]
    ):
        residual_items.append(
            {
                "sample_id": obs["sample_id"],
                "contact": obs["contact"],
                "residual_m": float(residual),
                "residual_mm": float(residual * 1000.0),
                "contact_point_in_base_m": [float(v) for v in point],
            }
        )

    return {
        "end_effector_fingertip_translation": {
            "right_outer_tip_in_ee_m": [float(v) for v in right],
            "left_outer_tip_in_ee_m": [float(v) for v in left],
            "right_outer_tip_in_ee_mm": [float(v * 1000.0) for v in right],
            "left_outer_tip_in_ee_mm": [float(v * 1000.0) for v in left],
            "outer_to_outer_distance_m": float(2.0 * abs(q[1])),
            "outer_to_outer_distance_mm": float(2.0 * abs(q[1]) * 1000.0),
        },
        "fitted_table_plane_in_base": {
            "equation": "normal[0]*x + normal[1]*y + normal[2]*z + d = 0",
            "normal": [float(v) for v in fit["plane_normal_base"]],
            "d_m": float(fit["plane_d_base_m"]),
        },
        "residual_summary": {
            "rms_residual_m": float(fit["rms_residual_m"]),
            "rms_residual_mm": float(fit["rms_residual_m"] * 1000.0),
            "max_abs_residual_m": float(fit["max_abs_residual_m"]),
            "max_abs_residual_mm": float(fit["max_abs_residual_m"] * 1000.0),
            "num_contact_observations": len(observations),
        },
        "trajectory_residuals": residual_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_path",
        type=Path,
        help="Input JSON containing the trajectories field.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional output JSON path. If omitted, only prints to stdout.",
    )
    args = parser.parse_args()

    observations = load_observations(args.json_path)
    fit = fit_fingertips(observations)
    output = build_output(observations, fit)

    print(json.dumps(output, indent=2))

    if args.output is not None:
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"\nSaved result JSON to: {args.output}")


if __name__ == "__main__":
    main()
