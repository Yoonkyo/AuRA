#!/usr/bin/env python3
r"""Fit one OAK camera's intrinsics (K, distortion) from a recorded ChArUco sweep.

Step 1 of the calibration chain Extract_intrinsic.py -> Extract_extrinsic.py ->
robot_world_calib_charuco.py; run it once per camera and per capture resolution.
The input is an aura/capture/record_3oak.py recording made while the board is
moved through the camera's field of view: <root>/bundle_XXXXXX/oak_<MXID>/rgb.jpg.
Views are filtered on marker count, ChArUco corner count, Laplacian blur, board
area and near-duplicate geometry and calibrated with
cv2.aruco.calibrateCameraCharucoExtended; when enough views exist the fit is
repeated without the worst --prune-worst-fraction of views and kept only if its
RMS is not higher.

Run from the repository root in the AURA_ENV conda environment with
config/aura.env loaded (set -a; source config/aura.env; set +a):

    python aura/calibration/Extract_intrinsic.py \
        --root "$AURA_DATA_ROOT/calibration_captures/<capture>" \
        --camera-name oak_<MXID> \
        --squares-x 4 --squares-y 5 --square-length 0.04 --marker-length 0.03 \
        --dictionary DICT_4X4_50 --sample-every 1 \
        --output-prefix "$AURA_CALIB_ROOT/intrinsic_results/1280x720/oak_<MXID>"

Writes <prefix>.npz (K (3, 3), dist (1, 5), rms (1,), image_size (2,) =
[width, height]) and <prefix>.json (the same values plus accepted views, filter
statistics, board, thresholds and per-view errors).

Notes:
  * Pass --sample-every 1. The default (30) keeps too few views of a short sweep
    and aborts with "Accepted only N valid views".
  * The pipeline reads $AURA_CALIB_ROOT/intrinsic_results/<width>x<height>/oak_<MXID>.{npz,json};
    the default --output-prefix (<root>/<camera-name>_charuco) is not that location.
  * Intrinsics are resolution specific: fit each resolution you record. AuRA records 1280x720
    ("high") by default; gamepad_record_franka_3oak.py defaults to "low" (pass --resolution high).
  * Uses the legacy cv2.aruco calibration API (detectMarkers,
    interpolateCornersCharuco, calibrateCameraCharucoExtended), available in the
    opencv-contrib-python 4.7 build of the AURA_ENV environment.
  * The JSON report records --root and the image paths exactly as given.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class AcceptedFrame:
    bundle_idx: int
    path: Path
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    marker_count: int
    charuco_count: int
    blur_score: float
    area_ratio: float
    center_xy: Tuple[float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Estimate intrinsics from one camera trajectory stored as bundle_XXXXXX/<camera_name>/rgb.jpg "
            "using OpenCV ChArUco detection and calibration."
        )
    )
    p.add_argument("--root", type=Path, required=True, help="Path containing bundle_000000, bundle_000001, ...")
    p.add_argument("--camera-name", type=str, required=True, help="Camera sub-directory inside each bundle: oak_<MXID> (one of $AURA_OAK_IDS with the oak_ prefix)")
    p.add_argument("--squares-x", type=int, required=True, help="Number of chessboard squares in X direction")
    p.add_argument("--squares-y", type=int, required=True, help="Number of chessboard squares in Y direction")
    p.add_argument("--square-length", type=float, required=True, help="Square side length, arbitrary unit if only intrinsics are needed")
    p.add_argument("--marker-length", type=float, required=True, help="Marker side length in same unit as square-length")
    p.add_argument("--dictionary", type=str, default="DICT_4X4_50", help="OpenCV aruco dictionary name")
    p.add_argument("--sample-every", type=int, default=30, help="Use every Nth bundle as a candidate before ChArUco filtering")
    p.add_argument("--max-candidates", type=int, default=0, help="0 means no limit after sampling")
    p.add_argument("--min-markers", type=int, default=4, help="Minimum detected marker count")
    p.add_argument("--min-charuco", type=int, default=8, help="Minimum interpolated ChArUco corner count")
    p.add_argument("--min-blur", type=float, default=60.0, help="Minimum variance of Laplacian threshold")
    p.add_argument("--min-area-ratio", type=float, default=0.03, help="Minimum board bbox area ratio in image")
    p.add_argument("--min-corner-shift", type=float, default=12.0, help="Dedup threshold on average common-corner shift in pixels")
    p.add_argument("--min-center-shift", type=float, default=24.0, help="Dedup threshold on board center shift in pixels")
    p.add_argument("--min-area-change", type=float, default=0.10, help="Dedup threshold on area ratio change")
    p.add_argument("--no-dedup", action="store_true", help="Disable geometry-based duplicate filtering")
    p.add_argument("--prune-worst-fraction", type=float, default=0.20, help="After first calibration, drop this fraction of worst views if Extended API is available and enough frames exist")
    p.add_argument("--min-views", type=int, default=10, help="Minimum accepted views required to calibrate")
    p.add_argument("--save-overlay-dir", type=Path, default=None, help="Optional directory for an overlay PNG of each candidate view it checks: bundle_XXXXXX_keep.png "
                   "if it passed the filters (the pruning re-fit may still drop it; accepted_bundle_indices in the JSON report lists the final views), else bundle_XXXXXX_drop.png")
    p.add_argument("--output-prefix", type=Path, default=None, help="Default: <root>/<camera-name>_charuco")
    return p.parse_args()


def get_aruco_dict(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build has no cv2.aruco module. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary name: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def create_charuco_board(squares_x: int, squares_y: int, square_length: float, marker_length: float, aruco_dict):
    # Prefer newer API, fall back to old create API.
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            return cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
        except Exception:
            pass
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)
    raise RuntimeError("Could not create CharucoBoard with this OpenCV version.")


def list_candidate_images(root: Path, camera_name: str, sample_every: int, max_candidates: int) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    for bundle_dir in sorted(root.glob("bundle_*")):
        try:
            bundle_idx = int(bundle_dir.name.split("_")[-1])
        except Exception:
            continue
        img_path = bundle_dir / camera_name / "rgb.jpg"
        if img_path.exists():
            items.append((bundle_idx, img_path))
    if sample_every > 1:
        items = items[::sample_every]
    if max_candidates > 0:
        items = items[:max_candidates]
    return items


def laplacian_blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_charuco(gray: np.ndarray, board, aruco_dict):
    # Compatibility path across OpenCV versions.
    params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None or len(ids) == 0:
        return None

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board,
    )
    if charuco_ids is None or charuco_corners is None:
        return None
    return {
        "marker_corners": corners,
        "marker_ids": ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
    }


def summarize_geometry(charuco_corners: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[float, Tuple[float, float]]:
    pts = charuco_corners.reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bbox_area = float(max(1.0, (x1 - x0) * (y1 - y0)))
    h, w = image_shape[:2]
    area_ratio = bbox_area / float(h * w)
    center = ((float(x0 + x1) * 0.5), (float(y0 + y1) * 0.5))
    return area_ratio, center


def mean_common_id_shift(prev: AcceptedFrame, cur_ids: np.ndarray, cur_corners: np.ndarray) -> float:
    prev_map = {int(i): p for i, p in zip(prev.charuco_ids.reshape(-1), prev.charuco_corners.reshape(-1, 2))}
    cur_map = {int(i): p for i, p in zip(cur_ids.reshape(-1), cur_corners.reshape(-1, 2))}
    common = sorted(set(prev_map.keys()) & set(cur_map.keys()))
    if len(common) < 4:
        return float("inf")
    shifts = [np.linalg.norm(cur_map[i] - prev_map[i]) for i in common]
    return float(np.mean(shifts))


def maybe_save_overlay(out_dir: Optional[Path], img_bgr: np.ndarray, det: dict, accepted: bool, bundle_idx: int):
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = img_bgr.copy()
    if det.get("marker_ids") is not None and len(det["marker_ids"]) > 0:
        cv2.aruco.drawDetectedMarkers(canvas, det["marker_corners"], det["marker_ids"])
    if det.get("charuco_ids") is not None and len(det["charuco_ids"]) > 0:
        cv2.aruco.drawDetectedCornersCharuco(canvas, det["charuco_corners"], det["charuco_ids"], (0, 255, 0) if accepted else (0, 0, 255))
    cv2.imwrite(str(out_dir / f"bundle_{bundle_idx:06d}_{'keep' if accepted else 'drop'}.png"), canvas)


def calibrate_charuco(board, accepted: Sequence[AcceptedFrame], image_size: Tuple[int, int]):
    charuco_corners = [x.charuco_corners for x in accepted]
    charuco_ids = [x.charuco_ids for x in accepted]

    if hasattr(cv2.aruco, "calibrateCameraCharucoExtended"):
        ret, K, dist, rvecs, tvecs, std_int, std_ext, per_view = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )
        return ret, K, dist, rvecs, tvecs, per_view.reshape(-1)

    ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )
    return ret, K, dist, rvecs, tvecs, None


def main() -> None:
    args = parse_args()
    aruco_dict = get_aruco_dict(args.dictionary)
    board = create_charuco_board(args.squares_x, args.squares_y, args.square_length, args.marker_length, aruco_dict)

    candidates = list_candidate_images(args.root, args.camera_name, args.sample_every, args.max_candidates)
    if not candidates:
        raise RuntimeError(f"No images found under {args.root}/bundle_*/{args.camera_name}/rgb.jpg")

    accepted: List[AcceptedFrame] = []
    image_size: Optional[Tuple[int, int]] = None
    stats = {
        "total_candidates": len(candidates),
        "accepted": 0,
        "rejected_no_board": 0,
        "rejected_too_few_markers": 0,
        "rejected_too_few_charuco": 0,
        "rejected_blur": 0,
        "rejected_small_board": 0,
        "rejected_duplicate": 0,
    }

    for bundle_idx, img_path in candidates:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        this_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = this_size
        elif image_size != this_size:
            # Skip inconsistent sizes; one trajectory should normally be fixed-size.
            continue

        det = detect_charuco(gray, board, aruco_dict)
        if det is None:
            stats["rejected_no_board"] += 1
            maybe_save_overlay(args.save_overlay_dir, img, {}, False, bundle_idx)
            continue

        marker_count = int(0 if det["marker_ids"] is None else len(det["marker_ids"]))
        charuco_count = int(0 if det["charuco_ids"] is None else len(det["charuco_ids"]))
        blur_score = laplacian_blur_score(gray)
        area_ratio, center = summarize_geometry(det["charuco_corners"], gray.shape)

        if marker_count < args.min_markers:
            stats["rejected_too_few_markers"] += 1
            maybe_save_overlay(args.save_overlay_dir, img, det, False, bundle_idx)
            continue
        if charuco_count < args.min_charuco:
            stats["rejected_too_few_charuco"] += 1
            maybe_save_overlay(args.save_overlay_dir, img, det, False, bundle_idx)
            continue
        if blur_score < args.min_blur:
            stats["rejected_blur"] += 1
            maybe_save_overlay(args.save_overlay_dir, img, det, False, bundle_idx)
            continue
        if area_ratio < args.min_area_ratio:
            stats["rejected_small_board"] += 1
            maybe_save_overlay(args.save_overlay_dir, img, det, False, bundle_idx)
            continue

        if accepted and not args.no_dedup:
            prev = accepted[-1]
            corner_shift = mean_common_id_shift(prev, det["charuco_ids"], det["charuco_corners"])
            center_shift = float(np.linalg.norm(np.array(center) - np.array(prev.center_xy)))
            area_change = abs(area_ratio - prev.area_ratio)
            too_similar = (
                corner_shift < args.min_corner_shift and
                center_shift < args.min_center_shift and
                area_change < args.min_area_change
            )
            if too_similar:
                stats["rejected_duplicate"] += 1
                maybe_save_overlay(args.save_overlay_dir, img, det, False, bundle_idx)
                continue

        accepted.append(
            AcceptedFrame(
                bundle_idx=bundle_idx,
                path=img_path,
                charuco_corners=det["charuco_corners"],
                charuco_ids=det["charuco_ids"],
                marker_count=marker_count,
                charuco_count=charuco_count,
                blur_score=blur_score,
                area_ratio=area_ratio,
                center_xy=center,
            )
        )
        stats["accepted"] += 1
        maybe_save_overlay(args.save_overlay_dir, img, det, True, bundle_idx)

    if image_size is None or len(accepted) < args.min_views:
        raise RuntimeError(
            f"Accepted only {len(accepted)} valid views, need at least {args.min_views}. "
            f"Try lowering thresholds or sampling more densely."
        )

    rms1, K1, dist1, rvecs1, tvecs1, per_view1 = calibrate_charuco(board, accepted, image_size)
    final_accepted = accepted
    final_rms, final_K, final_dist, per_view_final = rms1, K1, dist1, per_view1

    if per_view1 is not None and len(accepted) >= max(args.min_views + 5, 15) and 0.0 < args.prune_worst_fraction < 1.0:
        drop_n = int(len(accepted) * args.prune_worst_fraction)
        if drop_n > 0 and len(accepted) - drop_n >= args.min_views:
            ranked = np.argsort(per_view1)
            keep_idx = sorted(ranked[:-drop_n].tolist())
            pruned = [accepted[i] for i in keep_idx]
            rms2, K2, dist2, _, _, per_view2 = calibrate_charuco(board, pruned, image_size)
            if rms2 <= rms1:
                final_accepted = pruned
                final_rms, final_K, final_dist, per_view_final = rms2, K2, dist2, per_view2

    output_prefix = args.output_prefix or (args.root / f"{args.camera_name}_charuco")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "camera_name": args.camera_name,
        "root": str(args.root),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "rms_reprojection_error": float(final_rms),
        "K": np.asarray(final_K, dtype=float).tolist(),
        "dist": np.asarray(final_dist, dtype=float).reshape(-1).tolist(),
        "accepted_view_count": len(final_accepted),
        "accepted_bundle_indices": [x.bundle_idx for x in final_accepted],
        "accepted_image_paths": [str(x.path) for x in final_accepted],
        "filter_stats": stats,
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length": args.square_length,
            "marker_length": args.marker_length,
            "dictionary": args.dictionary,
        },
        "sampling": {
            "sample_every": args.sample_every,
            "max_candidates": args.max_candidates,
        },
        "thresholds": {
            "min_markers": args.min_markers,
            "min_charuco": args.min_charuco,
            "min_blur": args.min_blur,
            "min_area_ratio": args.min_area_ratio,
            "dedup_enabled": not args.no_dedup,
            "min_corner_shift": args.min_corner_shift,
            "min_center_shift": args.min_center_shift,
            "min_area_change": args.min_area_change,
            "prune_worst_fraction": args.prune_worst_fraction,
        },
    }
    if per_view_final is not None:
        report["per_view_errors"] = [float(x) for x in per_view_final.reshape(-1)]

    np.savez(
        str(output_prefix) + ".npz",
        K=np.asarray(final_K),
        dist=np.asarray(final_dist),
        rms=np.array([final_rms], dtype=np.float64),
        image_size=np.asarray(image_size, dtype=np.int32),
    )
    with open(str(output_prefix) + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"camera_name: {args.camera_name}")
    print(f"image_size : {image_size[0]} x {image_size[1]}")
    print(f"accepted   : {len(final_accepted)} / {stats['total_candidates']}")
    print(f"RMS error  : {final_rms:.6f}")
    print("K =")
    print(np.asarray(final_K))
    print("dist =")
    print(np.asarray(final_dist).reshape(-1))
    print(f"Saved: {output_prefix}.json")
    print(f"Saved: {output_prefix}.npz")
    print("=" * 60)


if __name__ == "__main__":
    main()
