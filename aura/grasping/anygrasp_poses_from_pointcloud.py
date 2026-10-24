#!/usr/bin/env python3
r"""
Run AnyGrasp on the fused observation point cloud (.npz or .ply) written by
aura/perception/phystwin_data_to_observation_pointcloud.py, export grasp
poses/scores, and optionally draw the exported grasps on the input cloud in
the two z-flipped debug views of the AnyGrasp SDK's grasp_detection/demo.py.

Expected .npz format:
    points: (N, 3)
    colors: (N, 3)

scripts/anygrasp_poses_one_shot.sh runs this as stage 03 (and, if that
yields no grasps, as stage 03b with --no_collision_detection). The AnyGrasp
SDK (https://github.com/graspnet/anygrasp_sdk), your license and
checkpoint_detection.tar are not part of this repository; set
AURA_ANYGRASP_SDK in config/aura.env to your checkout. Run inside the
$ANYGRASP_ENV conda env from the SDK's grasp_detection/ directory (it holds
license/, gsnet.so and lib_cxx.so), as the one-shot script does.

Example (the stage-03 arguments; <case> is a case directory under
$AURA_DATA_ROOT/franka_lift), from the repository root:
    set -a; source config/aura.env; set +a
    conda activate "$ANYGRASP_ENV"
    export PYTHONPATH="$AURA_ANYGRASP_SDK/grasp_detection${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cd "$AURA_ANYGRASP_SDK/grasp_detection"
    python "$AURA_REPO_ROOT/aura/grasping/anygrasp_poses_from_pointcloud.py" \
        --input "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp/observation/observation_frame000_filtered.npz" \
        --checkpoint_path "$AURA_ANYGRASP_SDK/grasp_detection/checkpoint_detection.tar" \
        --output_dir "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp_results" \
        --top_k 20 \
        --max_gripper_width 0.08 \
        --top_down_grasp \
        --save_debug_images

These exports stay in the shell, also after conda deactivate, so start robot
scripts and the lift one-shot from a new terminal (docs/hardware.md, Shell).

Outputs in --output_dir: grasps.json (rank 0 = best score after NMS),
best_grasp.json, result_summary.json, grasps_raw.npy and per-field
grasp_*.npy files. Poses are in meters, in the frame of the input cloud
(rotation column 0 = approach axis, column 1 = finger-closing axis). If no
grasp is found, only result_summary.json is written and the exit status is 0.
--max_gripper_width is clamped to [0, 0.1] m, as in the SDK demo. --debug sets
AnyGrasp's debug flag (as demo.py does), writes the same PNGs as
--save_debug_images, then opens two blocking Open3D windows in turn. The PNGs
render through a hidden Open3D window and need a display (a desktop session);
without one, Open3D can segfault after the grasp files are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input fused point cloud file (.npz or .ply)")
    parser.add_argument("--checkpoint_path", required=True, help="AnyGrasp checkpoint path")
    parser.add_argument("--output_dir", required=True, help="Directory to save poses/scores/visualizations")

    parser.add_argument("--max_gripper_width", type=float, default=0.1, help="Maximum gripper width (<=0.1m)")
    parser.add_argument("--gripper_height", type=float, default=0.03, help="Gripper height")
    parser.add_argument("--top_down_grasp", action="store_true", help="Output top-down grasps")

    parser.add_argument("--top_k", type=int, default=20, help="Number of top grasps to keep after NMS+sorting")
    parser.add_argument("--apply_object_mask", action="store_true", default=True,
                        help="Pass apply_object_mask=True to AnyGrasp (default: True)")
    parser.add_argument("--no_apply_object_mask", dest="apply_object_mask", action="store_false",
                        help="Disable apply_object_mask")
    parser.add_argument("--dense_grasp", action="store_true", help="Pass dense_grasp=True to AnyGrasp")
    parser.add_argument("--no_collision_detection", dest="collision_detection", action="store_false",
                        help="Disable collision detection")
    parser.set_defaults(collision_detection=True)

    parser.add_argument(
        "--lims",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Workspace limits. If omitted, infer from point cloud bounds with margin.",
    )
    parser.add_argument("--bounds_margin", type=float, default=0.02,
                        help="Margin (meters) added around inferred workspace bounds")

    parser.add_argument("--debug", action="store_true",
                        help="Open interactive Open3D windows like official demo.py")
    parser.add_argument("--save_debug_images", action="store_true",
                        help="Also save screenshots of [all grasps + cloud] and [best grasp + cloud]")
    parser.add_argument("--debug_flip_z", action="store_true", default=True,
                        help="Apply the same visualization-only z-flip used in the official demo (default: True)")
    parser.add_argument("--no_debug_flip_z", dest="debug_flip_z", action="store_false",
                        help="Disable the visualization-only z-flip")
    return parser


def load_cloud(input_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    suffix = input_path.suffix.lower()

    if suffix == ".npz":
        data = np.load(input_path)
        if "points" not in data or "colors" not in data:
            raise KeyError(f"{input_path} must contain keys 'points' and 'colors'.")
        points = np.asarray(data["points"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.float32)
    elif suffix == ".ply":
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(input_path))
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)
        if colors.size == 0:
            colors = np.ones_like(points, dtype=np.float32) * 0.5
    else:
        raise ValueError(f"Unsupported input format: {input_path}. Use .npz or .ply")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"colors must have shape (N, 3), got {colors.shape}")
    if len(points) != len(colors):
        raise ValueError(f"points and colors must have same length, got {len(points)} vs {len(colors)}")

    # Normalize colors into [0, 1] if necessary.
    if colors.max() > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)
    points = points.astype(np.float32)

    # Remove NaN / Inf rows.
    finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = points[finite_mask]
    colors = colors[finite_mask]

    if len(points) == 0:
        raise ValueError("No valid points remain after filtering NaN/Inf rows.")
    return points, colors


def infer_lims(points: np.ndarray, margin: float) -> List[float]:
    mins = points.min(axis=0) - margin
    maxs = points.max(axis=0) + margin
    return [float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]), float(mins[2]), float(maxs[2])]


class AnyGraspConfig(SimpleNamespace):
    checkpoint_path: str
    max_gripper_width: float
    gripper_height: float
    top_down_grasp: bool
    debug: bool


def make_cfg(args: argparse.Namespace) -> AnyGraspConfig:
    max_width = max(0.0, min(0.1, float(args.max_gripper_width)))
    return AnyGraspConfig(
        checkpoint_path=str(args.checkpoint_path),
        max_gripper_width=max_width,
        gripper_height=float(args.gripper_height),
        top_down_grasp=bool(args.top_down_grasp),
        debug=bool(args.debug),
    )


def to_pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def export_grasps(gg_pick, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw GraspGroup array (N, 17):
    # [score, width, height, depth, R(9), t(3), object_id]
    raw_path = output_dir / "grasps_raw.npy"
    gg_pick.save_npy(str(raw_path))

    scores = np.asarray(gg_pick.scores, dtype=np.float64)
    widths = np.asarray(gg_pick.widths, dtype=np.float64)
    heights = np.asarray(gg_pick.heights, dtype=np.float64)
    depths = np.asarray(gg_pick.depths, dtype=np.float64)
    rotations = np.asarray(gg_pick.rotation_matrices, dtype=np.float64)
    translations = np.asarray(gg_pick.translations, dtype=np.float64)
    object_ids = np.asarray(gg_pick.object_ids, dtype=np.int64)
    pose_matrices = np.stack([to_pose_matrix(r, t) for r, t in zip(rotations, translations)], axis=0)

    np.save(output_dir / "grasp_scores.npy", scores)
    np.save(output_dir / "grasp_widths.npy", widths)
    np.save(output_dir / "grasp_heights.npy", heights)
    np.save(output_dir / "grasp_depths.npy", depths)
    np.save(output_dir / "grasp_rotations.npy", rotations)
    np.save(output_dir / "grasp_translations.npy", translations)
    np.save(output_dir / "grasp_pose_matrices.npy", pose_matrices)
    np.save(output_dir / "grasp_object_ids.npy", object_ids)

    records = []
    for i in range(len(scores)):
        records.append(
            {
                "rank": int(i),
                "score": float(scores[i]),
                "width": float(widths[i]),
                "height": float(heights[i]),
                "depth": float(depths[i]),
                "translation": translations[i].tolist(),
                "rotation_matrix": rotations[i].tolist(),
                "pose_matrix_4x4": pose_matrices[i].tolist(),
                "object_id": int(object_ids[i]),
            }
        )

    with open(output_dir / "grasps.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    if records:
        with open(output_dir / "best_grasp.json", "w", encoding="utf-8") as f:
            json.dump(records[0], f, indent=2)


def build_o3d_cloud(points: np.ndarray, colors: np.ndarray):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return cloud


def maybe_flip_for_viz(cloud, grippers: list, enabled: bool):
    if not enabled:
        return cloud, grippers

    trans_mat = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, 1]],
        dtype=np.float64,
    )
    cloud.transform(trans_mat)
    for gripper in grippers:
        gripper.transform(trans_mat)
    return cloud, grippers


def interactive_debug_vis(cloud, grippers: list) -> None:
    import open3d as o3d

    o3d.visualization.draw_geometries([*grippers, cloud])
    if len(grippers) > 0:
        o3d.visualization.draw_geometries([grippers[0], cloud])


def save_o3d_screenshot(geometries: Sequence, out_path: Path, width: int = 1280, height: int = 960) -> None:
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="AnyGrasp Debug", width=width, height=height, visible=False)
    try:
        for geom in geometries:
            vis.add_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out_path), do_render=True)
    finally:
        vis.destroy_window()


def save_debug_visualizations(points: np.ndarray, colors: np.ndarray, gg_pick, output_dir: Path, flip_z: bool) -> None:
    cloud_all = build_o3d_cloud(points, colors)
    grippers_all = gg_pick.to_open3d_geometry_list()
    cloud_all, grippers_all = maybe_flip_for_viz(cloud_all, grippers_all, enabled=flip_z)

    if len(grippers_all) > 0:
        save_o3d_screenshot([*grippers_all, cloud_all], output_dir / "debug_all_grasps.png")
        save_o3d_screenshot([grippers_all[0], cloud_all], output_dir / "debug_best_grasp.png")
    else:
        save_o3d_screenshot([cloud_all], output_dir / "debug_cloud_only.png")


def print_summary(gg_pick) -> None:
    if len(gg_pick) == 0:
        print("No grasps after NMS/sorting.")
        return
    best = gg_pick[0]
    print("Top grasp scores:", np.asarray(gg_pick.scores))
    print("Best grasp score:", float(best.score))
    print("Best grasp translation [x, y, z]:", np.asarray(best.translation))
    print("Best grasp rotation matrix:\n", np.asarray(best.rotation_matrix))
    print("Best grasp width / height / depth:", float(best.width), float(best.height), float(best.depth))



def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points, colors = load_cloud(input_path)
    lims = list(map(float, args.lims)) if args.lims is not None else infer_lims(points, args.bounds_margin)
    print("Loaded point cloud:", points.shape, colors.shape)
    print("Workspace lims [xmin, xmax, ymin, ymax, zmin, zmax]:", lims)
    print("Point cloud bounds min/max:", points.min(axis=0), points.max(axis=0))

    from gsnet import AnyGrasp

    cfg = make_cfg(args)
    anygrasp = AnyGrasp(cfg)
    anygrasp.load_net()

    gg, _cloud_unused = anygrasp.get_grasp(
        points,
        colors,
        lims=lims,
        apply_object_mask=args.apply_object_mask,
        dense_grasp=args.dense_grasp,
        collision_detection=args.collision_detection,
    )

    if len(gg) == 0:
        print("No grasp detected after collision detection.")
        with open(output_dir / "result_summary.json", "w", encoding="utf-8") as f:
            json.dump({"num_grasps": 0, "lims": lims}, f, indent=2)
        return

    gg = gg.nms().sort_by_score()
    gg_pick = gg[: min(args.top_k, len(gg))]

    export_grasps(gg_pick, output_dir)
    print_summary(gg_pick)

    with open(output_dir / "result_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input": str(input_path),
                "num_input_points": int(len(points)),
                "num_grasps_after_nms": int(len(gg_pick)),
                "lims": lims,
                "best_score": float(gg_pick[0].score),
                "best_translation": np.asarray(gg_pick[0].translation).tolist(),
                "best_rotation_matrix": np.asarray(gg_pick[0].rotation_matrix).tolist(),
            },
            f,
            indent=2,
        )

    if args.save_debug_images or args.debug:
        try:
            save_debug_visualizations(points, colors, gg_pick, output_dir, flip_z=args.debug_flip_z)
            print("Saved debug images under:", output_dir)
        except Exception as e:
            print("Warning: failed to save debug images:", repr(e))

    if args.debug:
        cloud = build_o3d_cloud(points, colors)
        grippers = gg_pick.to_open3d_geometry_list()
        cloud, grippers = maybe_flip_for_viz(cloud, grippers, enabled=args.debug_flip_z)
        interactive_debug_vis(cloud, grippers)


if __name__ == "__main__":
    main()
