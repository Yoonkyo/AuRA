#!/usr/bin/env python3
r"""
Raw RGB-D + SAM/existing masks -> final fused observation point cloud.

Stage 02 of scripts/anygrasp_poses_one_shot.sh. Input is the static case
directory written by stage 01 (<case>/anygrasp: metadata.json, calibrate.pkl
with camera-to-world poses, color/<cam>/<frame>.png, depth/<cam>/<frame>.npy in
uint16 millimetres). For every camera in metadata.json the sibling
aura_segment_static_target_image.py (Grounded-SAM-2, same interpreter, model
files from --gsam-checkpoints) writes an RGBA mask; the pixels of its
--mask-object-id 8-connected component (default 0 = the largest) with valid
depth are lifted to the world frame, the --observation-cams are fused, and the
cloud is filtered: table cut -> voxel -> statistical -> radius -> PCA body crop.

Run in the AuRA environment with config/aura.env loaded
(set -a; source config/aura.env; set +a). The one-shot script calls:

python aura/perception/phystwin_data_to_observation_pointcloud.py \
  "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp" \
  --gsam-checkpoints "$AURA_GSAM_CHECKPOINTS" \
    --category "<object prompt>." \
    --frame 0 \
    --frames-for-pcd 0 \
    --observation-cams 0 1 2 \
    --mask-object-id 0 \
    --voxel-size 0.003 \
    --remove-statistical-outlier \
    --stat-nb-neighbors 35 \
    --stat-std-ratio 1.2 \
    --remove-radius-outlier \
    --radius-nb-points 10 \
    --radius 0.006 \
    --pca-body-filter \
    --pca-body-low 1.0 \
    --pca-body-high 99.0 \
    --pca-body-margin-long 0.0 \
    --pca-body-margin-cross 0.0 \
    --table-filter \
    --z-positive-down \
    --table-z 0.0 \
    --table-eps 0.01

The one-shot script derives --category from the case name (colorful_rope_6 ->
"colorful rope."). The file defaults differ from the values above (stat 20 /
2.0, radius 6 / 0.015, PCA 2-98 with margins 0.03 / 0.02), so a run without
these flags yields a differently filtered cloud. World z points DOWN (table at
z = 0): the table cut keeps z <= table_z + table_eps; --z-positive-down is
already the default and only --z-positive-up inverts the rule. --frames-for-pcd
is ignored and --repo-root is only recorded; both are kept for CLI compatibility.

Saved point clouds (the .npz, keys points / colors in [0, 1] / observation_cams,
is the input of aura/grasping/anygrasp_poses_from_pointcloud.py):
  observation/observation_frame000_filtered.ply
  observation/observation_frame000_filtered.npz

Saved masks:
  mask/<cam>/<instance_id>/<frame>.png
  mask/cam<cam>_frame<frame>_sam_rgba.png (+ .detections.jpg / .selection.json from the segmenter)
  mask/mask_info_<cam>.json
  mask/mask_info_all_cameras.json
  shape/masked_image.png (copy of the camera-0 RGBA; no v1 stage reads it)

Not saved:
  no mask/<cam>/all/<frame>.png
  no pcd/<frame>.npz
  no per-camera source point clouds
  no raw / before-pca / after-pca intermediate point-cloud files

Saved visualizations and metadata:
  observation/visualization/final_overlay_cam<cam>.png (one per observation camera)
  observation/visualization/topdown_final.png
  observation/visualization/topdown_filter_stages.png
  observation/visualization/observation_visualization_summary.json
  observation/raw_to_fused_observation_run_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d


# ----------------------------- utils ----------------------------- #

def _repo_root() -> Path:
    """Repository root: $AURA_REPO_ROOT, else two levels above aura/perception/."""
    return Path(os.environ.get("AURA_REPO_ROOT") or Path(__file__).resolve().parents[2])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    print("[CMD]", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, env=env, check=True)


def save_point_cloud(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    ensure_dir(path.parent)
    points = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is not None:
        colors = np.asarray(colors)
        if len(colors) == len(finite):
            colors = colors[finite]
        if len(colors) == len(points):
            colors = colors.astype(np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))

    o3d.io.write_point_cloud(str(path), pcd)


def load_depth(depth_path: Path, depth_scale: float) -> np.ndarray:
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth file: {depth_path}")

    if depth_path.suffix.lower() == ".npy":
        depth_raw = np.load(depth_path)
    else:
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Failed to read depth from {depth_path}")

    return depth_raw.astype(np.float32) / float(depth_scale)


def alpha_to_binary_mask(rgba_path: Path) -> np.ndarray:
    img = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read mask/RGBA image: {rgba_path}")
    if img.ndim == 2:
        return img > 0
    if img.shape[-1] == 4:
        return img[..., 3] > 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray > 0


def write_binary_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def split_binary_mask_components(mask: np.ndarray, min_area: int) -> List[np.ndarray]:
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    comps: List[Tuple[int, np.ndarray]] = []
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        comps.append((area, labels == label))

    comps.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in comps]


def validate_observation_cams(cams: Sequence[int], num_cam: int) -> List[int]:
    if len(cams) == 1 and int(cams[0]) == -1:
        return list(range(num_cam))
    out: List[int] = []
    for c in cams:
        c = int(c)
        if c < 0 or c >= num_cam:
            raise ValueError(f"Invalid camera index {c}; valid range is 0..{num_cam - 1}")
        if c not in out:
            out.append(c)
    if not out:
        raise ValueError("No observation cameras selected")
    return out


# ----------------------------- context ----------------------------- #

@dataclass
class CaseContext:
    repo_root: Path
    case_dir: Path
    mask_dir: Path
    observation_dir: Path
    metadata: dict
    intrinsics: np.ndarray
    c2ws: List[np.ndarray]
    frame_num: int
    num_cam: int


def build_context(repo_root: Path, case_dir: Path, output_subdir: str) -> CaseContext:
    metadata = read_json(case_dir / "metadata.json")
    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float32)

    with open(case_dir / "calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    c2ws = [np.asarray(x, dtype=np.float32) for x in c2ws]

    if len(c2ws) != len(intrinsics):
        raise ValueError(f"metadata has {len(intrinsics)} intrinsics, but calibrate.pkl has {len(c2ws)} poses")

    mask_dir = case_dir / "mask"
    observation_dir = case_dir / output_subdir
    ensure_dir(mask_dir)
    ensure_dir(observation_dir)

    return CaseContext(
        repo_root=repo_root,
        case_dir=case_dir,
        mask_dir=mask_dir,
        observation_dir=observation_dir,
        metadata=metadata,
        intrinsics=intrinsics,
        c2ws=c2ws,
        frame_num=int(metadata.get("frame_num", 1)),
        num_cam=len(intrinsics),
    )


# ----------------------------- masks ----------------------------- #

def save_component_masks_for_camera(
    ctx: CaseContext,
    cam: int,
    frame_indices: Sequence[int],
    category: str,
    union_mask: np.ndarray,
    components: Sequence[np.ndarray],
    min_area: int,
) -> Dict[str, object]:
    """Save only instance/component masks; do not save a mask/<cam>/all directory."""
    info: Dict[str, object] = {
        "category": category,
        "union_pixels": int(union_mask.sum()),
        "num_components": int(len(components)),
        "component_min_area": int(min_area),
        "components": {},
    }

    if len(components) == 0:
        components = [np.zeros_like(union_mask, dtype=bool)]

    for obj_id, comp in enumerate(components):
        for fidx in frame_indices:
            write_binary_mask(ctx.mask_dir / str(cam) / str(obj_id) / f"{fidx}.png", comp)
        info["components"][str(obj_id)] = {
            "num_pixels": int(comp.sum()),
            "rank_by_area": int(obj_id),
            "path_template": str(ctx.mask_dir / str(cam) / str(obj_id) / "<frame>.png"),
        }

    return info


def select_mask_for_fusion(mask_object_id: str, union_mask: np.ndarray, components: Sequence[np.ndarray]) -> np.ndarray:
    if mask_object_id in {"union", "all"}:
        return union_mask.astype(bool)

    if mask_object_id.isdigit():
        idx = int(mask_object_id)
        if idx < len(components):
            return components[idx].astype(bool)
        print(f"[WARN] requested component {idx}, but only {len(components)} components exist; using union")
        return union_mask.astype(bool)

    raise ValueError("--mask-object-id must be an integer component id, 'union', or 'all'")


def run_segmentation_for_static_case(
    ctx: CaseContext,
    category: str,
    frame_idx: int,
    mask_object_id: str,
    duplicate_mask_to_all_frames: bool,
    mask_component_min_area: int,
    gsam_checkpoints: Path,
) -> Dict[int, np.ndarray]:
    # AuRA static AnyGrasp segmentation is deliberately separate from
    # PhysTwin's video object/controller segmentation pipeline.
    segment_script = Path(__file__).resolve().parent / "aura_segment_static_target_image.py"
    if not segment_script.exists():
        raise FileNotFoundError(f"Missing {segment_script}")

    # The segmenter runs with cwd = its own directory: pass it the variables that
    # locate the Grounded-SAM-2 clone as absolute paths, resolving a ~ or a
    # relative value against this process's cwd.
    segment_env = dict(os.environ)
    for key in ("AURA_REPO_ROOT", "AURA_GSAM_REPO"):
        if segment_env.get(key):
            segment_env[key] = str(Path(segment_env[key]).expanduser().resolve())

    selected_masks: Dict[int, np.ndarray] = {}
    mask_info_all: Dict[str, object] = {}

    frame_indices = list(range(ctx.frame_num)) if duplicate_mask_to_all_frames else [frame_idx]

    for cam in range(ctx.num_cam):
        src = ctx.case_dir / "color" / str(cam) / f"{frame_idx}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing color image {src}")

        rgba_path = ctx.mask_dir / f"cam{cam}_frame{frame_idx}_sam_rgba.png"
        cmd = [
            sys.executable,
            str(segment_script),
            "--img_path", str(src),
            "--output_path", str(rgba_path),
            "--TEXT_PROMPT", category,
            "--checkpoint-dir", str(gsam_checkpoints),
        ]
        run_cmd(cmd, cwd=segment_script.parent, env=segment_env)

        union_mask = alpha_to_binary_mask(rgba_path)
        components = split_binary_mask_components(union_mask, mask_component_min_area)

        cam_info = save_component_masks_for_camera(
            ctx=ctx,
            cam=cam,
            frame_indices=frame_indices,
            category=category,
            union_mask=union_mask,
            components=components,
            min_area=mask_component_min_area,
        )

        selected = select_mask_for_fusion(mask_object_id, union_mask, components)
        selected_masks[cam] = selected

        cam_info["selected_for_fusion"] = mask_object_id
        cam_info["raw_sam_rgba"] = str(rgba_path)

        write_json(ctx.mask_dir / f"mask_info_{cam}.json", cam_info)
        mask_info_all[str(cam)] = cam_info

        print(
            f"[mask] cam {cam}: union_pixels={int(union_mask.sum())}, "
            f"components={len(components)}, selected={mask_object_id}, selected_pixels={int(selected.sum())}"
        )

        # Convenience copy of the camera-0 RGBA for inspection; no v1 stage reads it.
        if cam == 0:
            shape_dir = ctx.case_dir / "shape"
            ensure_dir(shape_dir)
            shutil.copy2(rgba_path, shape_dir / "masked_image.png")

    write_json(ctx.mask_dir / "mask_info_all_cameras.json", mask_info_all)
    return selected_masks


def load_existing_masks(ctx: CaseContext, frame_idx: int, mask_object_id: str) -> Dict[int, np.ndarray]:
    selected: Dict[int, np.ndarray] = {}

    for cam in range(ctx.num_cam):
        if mask_object_id in {"union", "all"}:
            cam_root = ctx.mask_dir / str(cam)
            if not cam_root.exists():
                raise FileNotFoundError(f"Missing mask directory {cam_root}")

            union: Optional[np.ndarray] = None
            used = 0
            for child in sorted(cam_root.iterdir(), key=lambda p: p.name):
                if not child.is_dir() or not child.name.isdigit():
                    continue
                mask_path = child / f"{frame_idx}.png"
                if not mask_path.exists():
                    continue
                img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                m = img > 0
                union = m if union is None else np.logical_or(union, m)
                used += 1
            if union is None:
                raise FileNotFoundError(f"No instance masks found under {cam_root} for frame {frame_idx}")
            selected[cam] = union
            print(f"[mask] cam {cam}: loaded union from {used} instance masks")
            continue

        mask_path = ctx.mask_dir / str(cam) / mask_object_id / f"{frame_idx}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing existing mask {mask_path}")
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read {mask_path}")
        selected[cam] = img > 0

    return selected


# ----------------------------- RGB-D lifting ----------------------------- #

def get_pcd_from_depth(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth_m.shape
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pix = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    pts = pix * depth_m.reshape(-1, 1)
    pts = pts @ np.linalg.inv(K).T
    return pts.reshape(h, w, 3).astype(np.float32)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3)
    homo = np.concatenate([flat, np.ones((len(flat), 1), dtype=flat.dtype)], axis=1)
    out = (T @ homo.T).T[:, :3]
    return out.reshape(points.shape).astype(np.float32)


def build_camera_world_points(
    ctx: CaseContext,
    frame_idx: int,
    cam: int,
    min_depth: float,
    max_depth: float,
    depth_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    color_path = ctx.case_dir / "color" / str(cam) / f"{frame_idx}.png"
    depth_path = ctx.case_dir / "depth" / str(cam) / f"{frame_idx}.npy"
    if not depth_path.exists():
        png_depth = ctx.case_dir / "depth" / str(cam) / f"{frame_idx}.png"
        if png_depth.exists():
            depth_path = png_depth

    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"Failed to read {color_path}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    depth_m = load_depth(depth_path, depth_scale)
    if depth_m.shape[:2] != color_rgb.shape[:2]:
        raise ValueError(f"Color/depth resolution mismatch for cam {cam}: color={color_rgb.shape[:2]}, depth={depth_m.shape[:2]}")

    points_cam = get_pcd_from_depth(depth_m, ctx.intrinsics[cam])
    valid = np.isfinite(points_cam).all(axis=-1)
    valid &= np.isfinite(depth_m)
    valid &= depth_m > float(min_depth)
    valid &= depth_m < float(max_depth)

    points_world = transform_points(points_cam, ctx.c2ws[cam])
    return points_world.astype(np.float32), color_rgb.astype(np.float32), valid.astype(bool)


# ----------------------------- filtering ----------------------------- #

def table_filter(
    points: np.ndarray,
    colors: np.ndarray,
    table_z: float,
    table_eps: float,
    z_positive_down: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    z = points[:, 2]
    if z_positive_down:
        keep = z <= table_z + table_eps
        rule = "keep z <= table_z + table_eps"
    else:
        keep = z >= table_z - table_eps
        rule = "keep z >= table_z - table_eps"

    return points[keep], colors[keep], {
        "enabled": True,
        "rule": rule,
        "before": int(len(points)),
        "after": int(keep.sum()),
        "table_z": float(table_z),
        "table_eps": float(table_eps),
        "z_positive_down": bool(z_positive_down),
        "z_percentiles_before": np.percentile(z, [0, 2, 50, 98, 100]).astype(float).tolist() if len(z) else None,
        "z_percentiles_after": np.percentile(points[keep, 2], [0, 2, 50, 98, 100]).astype(float).tolist() if keep.sum() else None,
    }


def open3d_filter_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    remove_statistical_outlier: bool,
    stat_nb_neighbors: int,
    stat_std_ratio: float,
    remove_radius_outlier: bool,
    radius_nb_points: int,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    finite = np.isfinite(points).all(axis=1)
    points = points[finite].astype(np.float32)
    colors = colors[finite].astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0

    meta: Dict[str, object] = {
        "initial_num_points": int(len(points)),
        "voxel_size": float(voxel_size),
        "remove_statistical_outlier": bool(remove_statistical_outlier),
        "stat_nb_neighbors": int(stat_nb_neighbors),
        "stat_std_ratio": float(stat_std_ratio),
        "remove_radius_outlier": bool(remove_radius_outlier),
        "radius_nb_points": int(radius_nb_points),
        "radius": float(radius),
        "stages": [],
    }

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))

    if voxel_size > 0 and len(pcd.points) > 0:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(float(voxel_size))
        meta["stages"].append({"name": "voxel_down_sample", "before": int(before), "after": int(len(pcd.points))})

    if remove_statistical_outlier and len(pcd.points) > stat_nb_neighbors:
        before = len(pcd.points)
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=int(stat_nb_neighbors),
            std_ratio=float(stat_std_ratio),
        )
        meta["stages"].append({"name": "remove_statistical_outlier", "before": int(before), "after": int(len(pcd.points)), "kept_indices": int(len(ind))})

    if remove_radius_outlier and len(pcd.points) > 0:
        before = len(pcd.points)
        pcd, ind = pcd.remove_radius_outlier(
            nb_points=int(radius_nb_points),
            radius=float(radius),
        )
        meta["stages"].append({"name": "remove_radius_outlier", "before": int(before), "after": int(len(pcd.points)), "kept_indices": int(len(ind))})

    out_points = np.asarray(pcd.points, dtype=np.float32)
    out_colors = np.asarray(pcd.colors, dtype=np.float32)
    meta["final_num_points"] = int(len(out_points))
    return out_points, out_colors, meta


def pca_body_filter(
    points: np.ndarray,
    colors: np.ndarray,
    low_percentile: float,
    high_percentile: float,
    margin_long: float,
    margin_cross: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)

    if len(points) < 10:
        return points, colors, {"enabled": True, "skipped": True, "reason": "too_few_points", "before": int(len(points)), "after": int(len(points))}

    if not (0.0 <= low_percentile < high_percentile <= 100.0):
        raise ValueError("Need 0 <= --pca-body-low < --pca-body-high <= 100")

    center = np.median(points, axis=0)
    X = points - center[None, :]
    _, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
    axes = Vt.T.astype(np.float32)
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0

    coords = X @ axes
    lo = np.percentile(coords, low_percentile, axis=0)
    hi = np.percentile(coords, high_percentile, axis=0)

    margins = np.array([margin_long, margin_cross, margin_cross], dtype=np.float32)
    lo2 = lo - margins
    hi2 = hi + margins

    keep = np.all((coords >= lo2[None, :]) & (coords <= hi2[None, :]), axis=1)

    return points[keep], colors[keep], {
        "enabled": True,
        "before": int(len(points)),
        "after": int(keep.sum()),
        "removed": int(len(points) - keep.sum()),
        "low_percentile": float(low_percentile),
        "high_percentile": float(high_percentile),
        "margin_long": float(margin_long),
        "margin_cross": float(margin_cross),
        "center_world_median": center.astype(float).tolist(),
        "axes_columns_world": axes.astype(float).tolist(),
        "singular_values": singular_values.astype(float).tolist(),
        "bounds_pca_before_margin": {"low": lo.astype(float).tolist(), "high": hi.astype(float).tolist()},
        "bounds_pca_after_margin": {"low": lo2.astype(float).tolist(), "high": hi2.astype(float).tolist()},
    }


# ----------------------------- fusion ----------------------------- #

def fuse_observation(
    ctx: CaseContext,
    frame_idx: int,
    selected_masks: Dict[int, np.ndarray],
    observation_cams: Sequence[int],
    min_depth: float,
    max_depth: float,
    depth_scale: float,
    table_filter_enabled: bool,
    table_z: float,
    table_eps: float,
    z_positive_down: bool,
    voxel_size: float,
    remove_statistical_outlier: bool,
    stat_nb_neighbors: int,
    stat_std_ratio: float,
    remove_radius_outlier: bool,
    radius_nb_points: int,
    radius: float,
    pca_body_filter_enabled: bool,
    pca_body_low: float,
    pca_body_high: float,
    pca_body_margin_long: float,
    pca_body_margin_cross: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    cams = validate_observation_cams(observation_cams, ctx.num_cam)

    obs_points: List[np.ndarray] = []
    obs_colors: List[np.ndarray] = []
    per_camera_stats: Dict[str, object] = {}

    for cam in cams:
        points_world, colors_rgb, depth_valid = build_camera_world_points(
            ctx,
            frame_idx=frame_idx,
            cam=cam,
            min_depth=min_depth,
            max_depth=max_depth,
            depth_scale=depth_scale,
        )

        obj_mask = np.logical_and(depth_valid, selected_masks[cam])
        cam_points = points_world[obj_mask]
        cam_colors = colors_rgb[obj_mask]

        finite = np.isfinite(cam_points).all(axis=1)
        cam_points = cam_points[finite]
        cam_colors = cam_colors[finite]

        obs_points.append(cam_points)
        obs_colors.append(cam_colors)

        per_camera_stats[str(cam)] = {
            "object_pixels": int(selected_masks[cam].sum()),
            "valid_object_points": int(len(cam_points)),
            "world_mean": cam_points.mean(axis=0).astype(float).tolist() if len(cam_points) else None,
            "world_min": cam_points.min(axis=0).astype(float).tolist() if len(cam_points) else None,
            "world_max": cam_points.max(axis=0).astype(float).tolist() if len(cam_points) else None,
        }

        print(f"[fuse] cam {cam}: object pixels={int(selected_masks[cam].sum())}, finite points={len(cam_points)}")

    if not obs_points:
        raise RuntimeError("No observation points were collected")

    raw_points = np.vstack(obs_points).astype(np.float32)
    raw_colors = np.vstack(obs_colors).astype(np.float32)

    stage_points: Dict[str, np.ndarray] = {"raw": raw_points.copy()}
    stage_colors: Dict[str, np.ndarray] = {"raw": raw_colors.copy()}

    filter_meta: Dict[str, object] = {
        "table_filter": {"enabled": False},
        "open3d_filters": {},
        "pca_body_filter": {"enabled": False},
    }

    points = raw_points
    colors = raw_colors

    if table_filter_enabled:
        points, colors, table_meta = table_filter(points, colors, table_z, table_eps, z_positive_down)
        filter_meta["table_filter"] = table_meta
        stage_points["after_table"] = points.copy()
        stage_colors["after_table"] = colors.copy()

    points, colors, open3d_meta = open3d_filter_cloud(
        points,
        colors,
        voxel_size=voxel_size,
        remove_statistical_outlier=remove_statistical_outlier,
        stat_nb_neighbors=stat_nb_neighbors,
        stat_std_ratio=stat_std_ratio,
        remove_radius_outlier=remove_radius_outlier,
        radius_nb_points=radius_nb_points,
        radius=radius,
    )
    filter_meta["open3d_filters"] = open3d_meta
    stage_points["before_pca"] = points.copy()
    stage_colors["before_pca"] = colors.copy()

    if pca_body_filter_enabled:
        points, colors, pca_meta = pca_body_filter(
            points,
            colors,
            low_percentile=pca_body_low,
            high_percentile=pca_body_high,
            margin_long=pca_body_margin_long,
            margin_cross=pca_body_margin_cross,
        )
        filter_meta["pca_body_filter"] = pca_meta
        stage_points["after_pca"] = points.copy()
        stage_colors["after_pca"] = colors.copy()

    stage_points["final"] = points.copy()
    stage_colors["final"] = colors.copy()

    meta = {
        "frame_idx": int(frame_idx),
        "observation_cams": cams,
        "raw_num_points": int(len(raw_points)),
        "filtered_num_points": int(len(points)),
        "per_camera_stats": per_camera_stats,
        "filter_meta": filter_meta,
    }

    print(f"[fuse] raw points={len(raw_points)} filtered points={len(points)}")
    return points.astype(np.float32), colors.astype(np.float32), stage_points, meta


# ----------------------------- visualization ----------------------------- #

def project_world(points_world: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float32)], axis=1)
    pts_c = (w2c @ pts_h.T).T[:, :3]
    z = pts_c[:, 2]

    valid = np.isfinite(pts_c).all(axis=1) & (z > 1e-6)
    uv = np.zeros((len(points_world), 2), dtype=np.float32)
    uv[:, 0] = K[0, 0] * (pts_c[:, 0] / z) + K[0, 2]
    uv[:, 1] = K[1, 1] * (pts_c[:, 1] / z) + K[1, 2]
    return uv, valid


def draw_projected_points(
    image_bgr: np.ndarray,
    points_world: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    color_bgr: Tuple[int, int, int],
    radius: int,
    max_draw: int,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    uv, valid = project_world(points_world, K, c2w)
    in_img = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    idx = np.where(in_img)[0]
    if len(idx) > max_draw:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_draw, replace=False)

    for i in idx:
        x = int(round(float(uv[i, 0])))
        y = int(round(float(uv[i, 1])))
        cv2.circle(out, (x, y), radius, color_bgr, -1, lineType=cv2.LINE_AA)

    return out


def _subsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def _xy_limits(stage_points: Dict[str, np.ndarray], margin: float = 0.02) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    pts = np.vstack([p for p in stage_points.values() if p is not None and len(p)])
    xmin, ymin = np.min(pts[:, :2], axis=0)
    xmax, ymax = np.max(pts[:, :2], axis=0)
    return (float(xmin - margin), float(xmax + margin)), (float(ymin - margin), float(ymax + margin))


def write_topdown_plot(
    out_path: Path,
    stage_points: Dict[str, np.ndarray],
    order: Sequence[str],
    title: str,
    max_points: int,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(out_path.parent)
    plt.figure(figsize=(7, 7))
    xlim, ylim = _xy_limits({k: stage_points[k] for k in order if k in stage_points})

    for i, name in enumerate(order):
        if name not in stage_points:
            continue
        pts = _subsample(stage_points[name], max_points, seed=10 + i)
        plt.scatter(pts[:, 0], pts[:, 1], s=1, label=f"{name} ({len(stage_points[name])})")

    plt.axis("equal")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("world x")
    plt.ylabel("world y")
    plt.title(title)
    plt.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_visualizations(
    ctx: CaseContext,
    frame_idx: int,
    observation_cams: Sequence[int],
    stage_points: Dict[str, np.ndarray],
    projection_max_draw: int,
    topdown_max_points: int,
) -> List[str]:
    vis_dir = ctx.observation_dir / "visualization"
    ensure_dir(vis_dir)

    outputs: List[str] = []
    final_points = stage_points["final"]

    # final overlay for each view.
    for cam in validate_observation_cams(observation_cams, ctx.num_cam):
        img_path = ctx.case_dir / "color" / str(cam) / f"{frame_idx}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] cannot read {img_path}; skip visualization for cam {cam}")
            continue

        overlay = draw_projected_points(
            img,
            final_points,
            ctx.intrinsics[cam],
            ctx.c2ws[cam],
            color_bgr=(0, 0, 255),
            radius=1,
            max_draw=projection_max_draw,
        )
        out = vis_dir / f"final_overlay_cam{cam}.png"
        cv2.imwrite(str(out), overlay)
        outputs.append(str(out))

    # top-down final.
    out_final = vis_dir / "topdown_final.png"
    write_topdown_plot(
        out_final,
        {"final": final_points},
        ["final"],
        "Top-down XY: final filtered observation cloud",
        topdown_max_points,
    )
    outputs.append(str(out_final))

    # top-down all stages, but no stage point clouds are saved to disk.
    stage_order = [s for s in ["raw", "after_table", "before_pca", "after_pca", "final"] if s in stage_points]
    out_stages = vis_dir / "topdown_filter_stages.png"
    write_topdown_plot(
        out_stages,
        stage_points,
        stage_order,
        "Top-down XY: filtering stages",
        topdown_max_points,
    )
    outputs.append(str(out_stages))

    summary = {
        "visualization_dir": str(vis_dir),
        "outputs": outputs,
        "stage_point_counts": {k: int(len(v)) for k, v in stage_points.items()},
        "notes": {
            "final_overlay_cam*.png": "Final filtered cloud projected to each RGB view in red",
            "topdown_final.png": "Final filtered cloud only",
            "topdown_filter_stages.png": "Stage comparison generated from in-memory clouds; intermediate point clouds are not saved",
        },
    }
    summary_path = vis_dir / "observation_visualization_summary.json"
    write_json(summary_path, summary)
    outputs.append(str(summary_path))
    return outputs


# ----------------------------- CLI ----------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw RGB-D + masks/SAM -> final fused observation point cloud. No shape prior.")

    parser.add_argument("case_dir", type=str)
    parser.add_argument("--repo-root", type=str, default=".", help="Unused; kept for CLI compatibility and recorded in the run metadata")
    parser.add_argument("--gsam-checkpoints", type=str, default=os.environ.get("AURA_GSAM_CHECKPOINTS") or str(_repo_root() / "checkpoints" / "groundedSAM"), help="Grounded-SAM-2 model directory, forwarded to the segmenter as --checkpoint-dir. Default: $AURA_GSAM_CHECKPOINTS, else <repo>/checkpoints/groundedSAM")
    parser.add_argument("--category", type=str, default="rope.", help="GroundingDINO/SAM prompt. Default: rope.")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--frames-for-pcd", type=str, default="0", help="Kept for CLI compatibility; this compact script only processes --frame")
    parser.add_argument("--output-subdir", type=str, default="observation")

    parser.add_argument("--skip-sam", action="store_true")
    parser.add_argument("--mask-object-id", type=str, default="0", help="Component id for fusion. Default: 0 = largest component. Use 'union' to fuse all saved/segmented components without saving mask/all.")
    parser.add_argument("--duplicate-mask-to-all-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-component-min-area", type=int, default=20)

    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=1.5)
    parser.add_argument("--depth-scale", type=float, default=1000.0)

    parser.add_argument("--observation-cams", nargs="+", type=int, default=[0, 1, 2])

    parser.add_argument("--table-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--table-z", type=float, default=0.0)
    parser.add_argument("--table-eps", type=float, default=0.01)
    parser.add_argument("--z-positive-down", action="store_true", default=True, help="Default: world z points down and the table cut keeps z <= table_z + table_eps; --z-positive-up inverts the rule.")
    parser.add_argument("--z-positive-up", dest="z_positive_down", action="store_false")

    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument("--remove-statistical-outlier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stat-nb-neighbors", type=int, default=20)
    parser.add_argument("--stat-std-ratio", type=float, default=2.0)

    parser.add_argument("--remove-radius-outlier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radius-nb-points", type=int, default=6)
    parser.add_argument("--radius", type=float, default=0.015)

    parser.add_argument("--pca-body-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca-body-low", type=float, default=2.0)
    parser.add_argument("--pca-body-high", type=float, default=98.0)
    parser.add_argument("--pca-body-margin-long", type=float, default=0.03)
    parser.add_argument("--pca-body-margin-cross", type=float, default=0.02)

    parser.add_argument("--write-visualization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vis-max-draw", type=int, default=8000)
    parser.add_argument("--vis-topdown-max-points", type=int, default=12000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    case_dir = Path(args.case_dir).resolve()
    repo_root = Path(args.repo_root).resolve()
    ctx = build_context(repo_root, case_dir, args.output_subdir)

    if args.skip_sam:
        selected_masks = load_existing_masks(ctx, frame_idx=args.frame, mask_object_id=args.mask_object_id)
    else:
        if not args.category:
            raise ValueError("--category is required unless --skip-sam is used")
        selected_masks = run_segmentation_for_static_case(
            ctx,
            category=args.category,
            frame_idx=args.frame,
            mask_object_id=args.mask_object_id,
            duplicate_mask_to_all_frames=args.duplicate_mask_to_all_frames,
            mask_component_min_area=args.mask_component_min_area,
            gsam_checkpoints=Path(args.gsam_checkpoints).expanduser().resolve(),
        )

    final_points, final_colors, stage_points, fusion_meta = fuse_observation(
        ctx=ctx,
        frame_idx=args.frame,
        selected_masks=selected_masks,
        observation_cams=args.observation_cams,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        depth_scale=args.depth_scale,
        table_filter_enabled=args.table_filter,
        table_z=args.table_z,
        table_eps=args.table_eps,
        z_positive_down=args.z_positive_down,
        voxel_size=args.voxel_size,
        remove_statistical_outlier=args.remove_statistical_outlier,
        stat_nb_neighbors=args.stat_nb_neighbors,
        stat_std_ratio=args.stat_std_ratio,
        remove_radius_outlier=args.remove_radius_outlier,
        radius_nb_points=args.radius_nb_points,
        radius=args.radius,
        pca_body_filter_enabled=args.pca_body_filter,
        pca_body_low=args.pca_body_low,
        pca_body_high=args.pca_body_high,
        pca_body_margin_long=args.pca_body_margin_long,
        pca_body_margin_cross=args.pca_body_margin_cross,
    )

    final_ply = ctx.observation_dir / f"observation_frame{args.frame:03d}_filtered.ply"
    final_npz = ctx.observation_dir / f"observation_frame{args.frame:03d}_filtered.npz"

    save_point_cloud(final_ply, final_points, final_colors)
    np.savez(
        final_npz,
        points=final_points.astype(np.float32),
        colors=final_colors.astype(np.float32),
        observation_cams=np.asarray(validate_observation_cams(args.observation_cams, ctx.num_cam), dtype=np.int32),
    )

    visualization_outputs: List[str] = []
    if args.write_visualization:
        visualization_outputs = write_visualizations(
            ctx=ctx,
            frame_idx=args.frame,
            observation_cams=args.observation_cams,
            stage_points=stage_points,
            projection_max_draw=args.vis_max_draw,
            topdown_max_points=args.vis_topdown_max_points,
        )

    run_meta = {
        "case_dir": str(case_dir),
        "repo_root": str(repo_root),
        "frame": int(args.frame),
        "category": args.category,
        "mask_object_id": args.mask_object_id,
        "only_final_pointcloud_saved": True,
        "no_mask_all_saved": True,
        "final_outputs": {
            "ply": str(final_ply),
            "npz": str(final_npz),
        },
        "fusion_meta": fusion_meta,
        "visualization_outputs": visualization_outputs,
    }
    meta_path = ctx.observation_dir / "raw_to_fused_observation_run_meta.json"
    write_json(meta_path, run_meta)

    print(json.dumps({
        "status": "ok",
        "filtered_points": int(len(final_points)),
        "final_ply": str(final_ply),
        "final_npz": str(final_npz),
        "visualization_dir": str(ctx.observation_dir / "visualization") if args.write_visualization else None,
        "run_meta": str(meta_path),
    }, indent=2))


if __name__ == "__main__":
    main()
