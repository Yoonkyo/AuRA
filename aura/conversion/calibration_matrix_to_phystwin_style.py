#!/usr/bin/env python3
r"""Write the PhysTwin-style calibrate.pkl and metadata.json for a converted case.

Run after aura/conversion/raw_data_to_phystwin_style.py, on the same directory.
For every camera in --camera-order one intrinsic file is looked up in
--intrinsic-dir and one extrinsic file in --extrinsic-dir: the oak_<MXID>.json
and oak_<MXID>.npz files written by aura/calibration/Extract_intrinsic.py and
aura/calibration/Extract_extrinsic.py. A JSON file is matched by its
camera_name key first, then by file name; an NPZ file by file name only.

Outputs in --output-dir:
    calibrate.pkl   pickled list, in --camera-order, of 4x4 camera-to-world
                    matrices (nested float lists; phys_twin_c2w, else
                    T_world_from_cam)
    metadata.json   camera_names, serial_numbers, frame_num, intrinsics (3x3 K
                    per camera; --crop shifts the principal point), WH (shared
                    [w, h]; null if the cameras differ, an error with
                    --force-single-wh) and per_camera (source files, sizes,
                    crop, dist, c2w)

Pitfalls:
  - metadata.json REPLACES the conversion record written by
    raw_data_to_phystwin_style.py (whose frame count is used when --frame-num
    is omitted): copy that file to conversion_metadata.json right after the
    raw conversion (as its Example does), not before a re-run of this script.
  - --camera-order must be the order of color/<i> and depth/<i>, i.e. the
    --camera-order given to raw_data_to_phystwin_style.py.
  - --crop CAMERA_NAME must match a --camera-order entry exactly; any other
    name is ignored without a warning (K and WH stay uncropped): pass the
    same --crop options to both converters.
  - --intrinsic-dir must match the capture resolution
    ($AURA_CALIB_ROOT/intrinsic_results/1280x720 or .../640x400).
  - World = ChArUco board frame with z pointing down (cameras above the table
    have negative z); robot_calib_result.npz (T_world_from_base) uses the same
    frame. Do not pass --make-z-positive in the AuRA pipeline: it rotates the
    world frame and breaks that consistency.
  - The NPZ fallback loads with allow_pickle=True: use trusted files only.

Example
-------
# from the repository root, after raw_data_to_phystwin_style.py (the arguments
# of stage 04 of scripts/run_anygrasp_to_lift_one_shot.sh)
set -a; source config/aura.env; set +a
conda activate "$AURA_ENV"
CASE_DIR="$AURA_DATA_ROOT/franka_lift/<case>/phystwin"
CAMERA_ORDER=$(printf 'oak_%s ' $AURA_OAK_IDS)
# conversion_metadata.json is copied right after raw_data_to_phystwin_style.py;
# do not copy again on a re-run: metadata.json is this script's output by then
N=$(find "$CASE_DIR/color/0" -maxdepth 1 -name '*.png' | wc -l)
python aura/conversion/calibration_matrix_to_phystwin_style.py \
  --intrinsic-dir "$AURA_CALIB_ROOT/intrinsic_results/1280x720" \
  --extrinsic-dir "$AURA_CALIB_ROOT/extrinsic_results" \
  --output-dir "$CASE_DIR" \
  --camera-order $CAMERA_ORDER --serial-numbers $CAMERA_ORDER \
  --frame-num "$N" --force-single-wh --overwrite
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Helpers
# -----------------------------

def normalize_name(name: str) -> str:
    return name.strip().lower()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def is_intrinsic_json(d: dict) -> bool:
    return "K" in d and "dist" in d and (
        ("image_width" in d and "image_height" in d) or ("image_size" in d)
    ) and "phys_twin_c2w" not in d


def is_extrinsic_json(d: dict) -> bool:
    return "phys_twin_c2w" in d or "T_world_from_cam" in d


def find_matching_file_by_camera_name(
    folder: Path,
    camera_name: str,
    kind: str,
) -> Tuple[Path, dict]:
    """
    Prefer JSON because it carries explicit keys like camera_name / phys_twin_c2w.
    Falls back to NPZ matched by filename only.
    kind: 'intrinsic' or 'extrinsic'
    """
    camera_key = normalize_name(camera_name)

    # 1) JSONs: robust match by file content first, then filename.
    json_candidates = sorted(folder.glob("*.json"))
    content_matches: List[Tuple[Path, dict]] = []
    filename_matches: List[Tuple[Path, dict]] = []
    for p in json_candidates:
        try:
            d = load_json(p)
        except Exception:
            continue

        if kind == "intrinsic" and not is_intrinsic_json(d):
            continue
        if kind == "extrinsic" and not is_extrinsic_json(d):
            continue

        data_camera_name = normalize_name(str(d.get("camera_name", "")))
        if data_camera_name == camera_key:
            content_matches.append((p, d))
        elif camera_key in normalize_name(p.name):
            filename_matches.append((p, d))

    if len(content_matches) == 1:
        return content_matches[0]
    if len(content_matches) > 1:
        raise RuntimeError(
            f"Found multiple {kind} JSON files for camera '{camera_name}' in {folder}:\n"
            + "\n".join(str(p) for p, _ in content_matches)
        )
    if len(filename_matches) == 1:
        return filename_matches[0]
    if len(filename_matches) > 1:
        raise RuntimeError(
            f"Found multiple filename-matched {kind} JSON files for camera '{camera_name}' in {folder}:\n"
            + "\n".join(str(p) for p, _ in filename_matches)
        )

    # 2) NPZ fallback: filename-based only.
    npz_candidates = sorted(folder.glob("*.npz"))
    npz_matches = [p for p in npz_candidates if camera_key in normalize_name(p.name)]
    if len(npz_matches) == 1:
        return npz_matches[0], load_npz(npz_matches[0])
    if len(npz_matches) > 1:
        raise RuntimeError(
            f"Found multiple filename-matched {kind} NPZ files for camera '{camera_name}' in {folder}:\n"
            + "\n".join(str(p) for p in npz_matches)
        )

    raise FileNotFoundError(
        f"Could not find an {kind} file for camera '{camera_name}' in {folder}."
    )


def parse_crop_args(crop_args: Optional[List[List[str]]]) -> Dict[str, Tuple[int, int, int, int]]:
    crops: Dict[str, Tuple[int, int, int, int]] = {}
    if not crop_args:
        return crops
    for item in crop_args:
        if len(item) != 5:
            raise ValueError(
                "Each --crop must be: --crop CAMERA_NAME TOP BOTTOM LEFT RIGHT"
            )
        cam = item[0]
        top, bottom, left, right = map(int, item[1:])
        if min(top, bottom, left, right) < 0:
            raise ValueError(f"Crop values must be >= 0 for camera '{cam}'.")
        crops[cam] = (top, bottom, left, right)
    return crops


def apply_crop_to_K(K: np.ndarray, crop: Tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = crop
    K_new = np.array(K, dtype=np.float64, copy=True)
    K_new[0, 2] -= left
    K_new[1, 2] -= top
    return K_new


def compute_output_wh(
    input_wh: Tuple[int, int], crop: Tuple[int, int, int, int]
) -> Tuple[int, int]:
    w, h = input_wh
    top, bottom, left, right = crop
    new_w = w - left - right
    new_h = h - top - bottom
    if new_w <= 0 or new_h <= 0:
        raise ValueError(
            f"Invalid crop {crop} for input size {(w, h)} -> output {(new_w, new_h)}"
        )
    return new_w, new_h


def to_float_nested_list(arr) -> list:
    return np.asarray(arr, dtype=np.float64).tolist()


def make_world_fix_matrix(axis: str = "x") -> np.ndarray:
    """
    Build a proper rotation (det=+1) that flips the sign of Z while keeping
    relative camera geometry unchanged.

    axis='x' -> 180 degree rotation about X: diag([1, -1, -1])
    axis='y' -> 180 degree rotation about Y: diag([-1, 1, -1])

    This is a global rigid transform applied equally to all cameras:
        c2w_new = T_fix @ c2w_old
    so pairwise camera distances and relative translation are preserved.
    """
    if axis == "x":
        R = np.diag([1.0, -1.0, -1.0])
    elif axis == "y":
        R = np.diag([-1.0, 1.0, -1.0])
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def infer_frame_num_from_output_dir(output_dir: Path) -> Optional[int]:
    """
    Try to infer the number of frames from the existing converted dataset.

    Priority:
    1) An existing metadata.json in output_dir: its frame_num (the PhysTwin
       key this script writes, e.g. on an earlier run), else its num_frames
       (written by raw_data_to_phystwin_style.py)
    2) Count PNGs under color/0
    """
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        try:
            existing = load_json(metadata_path)
            if "frame_num" in existing and existing["frame_num"] is not None:
                return int(existing["frame_num"])
            if "num_frames" in existing and existing["num_frames"] is not None:
                return int(existing["num_frames"])
        except Exception:
            pass

    color0 = output_dir / "color" / "0"
    if color0.is_dir():
        return len(list(color0.glob("*.png")))

    return None


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PhysTwin calibrate.pkl and metadata.json from intrinsic/extrinsic folders."
    )
    parser.add_argument("--intrinsic-dir", type=Path, required=True)
    parser.add_argument("--extrinsic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--camera-order",
        nargs="+",
        required=True,
        help="Camera names in the exact order used by color/0,1,2... and depth/0,1,2...",
    )
    parser.add_argument(
        "--crop",
        nargs=5,
        metavar=("CAMERA_NAME", "TOP", "BOTTOM", "LEFT", "RIGHT"),
        action="append",
        help=(
            "Optional crop to apply to the intrinsic principal point. May be repeated. "
            "CAMERA_NAME must match a --camera-order entry exactly (case included); any other "
            "name is ignored without a warning, whereas raw_data_to_phystwin_style.py rejects it."
        ),
    )
    parser.add_argument(
        "--serial-numbers",
        nargs="+",
        default=None,
        help="Optional serial numbers to write into metadata.json. Defaults to --camera-order.",
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        default=None,
        help="Optional frame count to write into metadata.json. If omitted, the script will try to infer it.",
    )
    parser.add_argument(
        "--make-z-positive",
        action="store_true",
        help=(
            "Apply a global 180-degree rotation to all c2w matrices so that Z changes sign "
            "while preserving relative camera translation. Not for the AuRA pipeline: "
            "robot_calib_result.npz expects the unrotated (z-down) world frame."
        ),
    )
    parser.add_argument(
        "--z-flip-axis",
        choices=["x", "y"],
        default="x",
        help="Axis used for the 180-degree world rotation when --make-z-positive is enabled (default: x).",
    )
    parser.add_argument(
        "--force-single-wh",
        action="store_true",
        help="Require all cameras to share the same output WH and write a single global WH.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing calibrate.pkl / metadata.json.",
    )

    args = parser.parse_args()

    args.intrinsic_dir = args.intrinsic_dir.resolve()
    args.extrinsic_dir = args.extrinsic_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.intrinsic_dir.is_dir():
        raise NotADirectoryError(f"Intrinsic dir not found: {args.intrinsic_dir}")
    if not args.extrinsic_dir.is_dir():
        raise NotADirectoryError(f"Extrinsic dir not found: {args.extrinsic_dir}")

    if args.serial_numbers is not None and len(args.serial_numbers) != len(args.camera_order):
        raise ValueError(
            "--serial-numbers must have the same length as --camera-order"
        )

    crops = parse_crop_args(args.crop)

    calibrate_path = args.output_dir / "calibrate.pkl"
    metadata_path = args.output_dir / "metadata.json"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        for p in [calibrate_path, metadata_path]:
            if p.exists():
                raise FileExistsError(
                    f"Output file already exists: {p}. Use --overwrite to replace it."
                )

    world_fix = make_world_fix_matrix(args.z_flip_axis) if args.make_z_positive else np.eye(4, dtype=np.float64)

    intrinsics_out: List[list] = []
    c2ws_out: List[list] = []
    widths_heights: List[Tuple[int, int]] = []
    per_camera_meta: List[dict] = []

    for cam_name in args.camera_order:
        intr_path, intr = find_matching_file_by_camera_name(args.intrinsic_dir, cam_name, "intrinsic")
        extr_path, extr = find_matching_file_by_camera_name(args.extrinsic_dir, cam_name, "extrinsic")

        K = np.asarray(intr["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"Intrinsic K for camera '{cam_name}' is not 3x3: {K.shape}")

        dist = np.asarray(intr.get("dist", []), dtype=np.float64).reshape(-1)

        if "image_width" in intr and "image_height" in intr:
            input_wh = (int(intr["image_width"]), int(intr["image_height"]))
        elif "image_size" in intr:
            image_size = np.asarray(intr["image_size"]).reshape(-1)
            input_wh = (int(image_size[0]), int(image_size[1]))
        else:
            raise ValueError(
                f"Intrinsic file for camera '{cam_name}' has no image size fields: {intr_path}"
            )

        crop = crops.get(cam_name, (0, 0, 0, 0))
        output_wh = compute_output_wh(input_wh, crop)
        K_out = apply_crop_to_K(K, crop)

        if "phys_twin_c2w" in extr:
            c2w = np.asarray(extr["phys_twin_c2w"], dtype=np.float64)
        elif "T_world_from_cam" in extr:
            c2w = np.asarray(extr["T_world_from_cam"], dtype=np.float64)
        else:
            raise ValueError(
                f"Extrinsic file for camera '{cam_name}' has neither phys_twin_c2w nor T_world_from_cam: {extr_path}"
            )
        if c2w.shape != (4, 4):
            raise ValueError(f"c2w for camera '{cam_name}' is not 4x4: {c2w.shape}")

        c2w_out = world_fix @ c2w

        intrinsics_out.append(to_float_nested_list(K_out))
        c2ws_out.append(to_float_nested_list(c2w_out))
        widths_heights.append(output_wh)

        per_camera_meta.append(
            {
                "camera_name": cam_name,
                "intrinsic_source": str(intr_path),
                "extrinsic_source": str(extr_path),
                "input_WH": [int(input_wh[0]), int(input_wh[1])],
                "output_WH": [int(output_wh[0]), int(output_wh[1])],
                "crop": {
                    "top": int(crop[0]),
                    "bottom": int(crop[1]),
                    "left": int(crop[2]),
                    "right": int(crop[3]),
                },
                "dist": dist.astype(np.float64).tolist(),
                "z_positive_fix_applied": bool(args.make_z_positive),
                "z_positive_fix_axis": args.z_flip_axis if args.make_z_positive else None,
                "phys_twin_c2w_original": to_float_nested_list(c2w),
                "phys_twin_c2w_written": to_float_nested_list(c2w_out),
            }
        )

    unique_wh = sorted(set(widths_heights))
    serial_numbers = list(args.serial_numbers) if args.serial_numbers is not None else list(args.camera_order)
    frame_num = args.frame_num if args.frame_num is not None else infer_frame_num_from_output_dir(args.output_dir)

    if args.force_single_wh:
        if len(unique_wh) != 1:
            raise ValueError(
                "Cameras do not share the same output WH after cropping: "
                + ", ".join(str(x) for x in unique_wh)
            )
        metadata = {
            "camera_names": list(args.camera_order),
            "serial_numbers": serial_numbers,
            "frame_num": frame_num,
            "intrinsics": intrinsics_out,
            "WH": [int(unique_wh[0][0]), int(unique_wh[0][1])],
            "per_camera": per_camera_meta,
        }
    else:
        metadata = {
            "camera_names": list(args.camera_order),
            "serial_numbers": serial_numbers,
            "frame_num": frame_num,
            "intrinsics": intrinsics_out,
            "WH": [int(unique_wh[0][0]), int(unique_wh[0][1])] if len(unique_wh) == 1 else None,
            "per_camera": per_camera_meta,
        }

    with open(calibrate_path, "wb") as f:
        pickle.dump(c2ws_out, f)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 72)
    print(f"Saved calibrate.pkl : {calibrate_path}")
    print(f"Saved metadata.json : {metadata_path}")
    print("Camera order:")
    for idx, cam in enumerate(args.camera_order):
        wh = widths_heights[idx]
        z_translation = c2ws_out[idx][2][3]
        print(f"  [{idx}] {cam} -> WH={wh}, written c2w tz={z_translation:.6f}")
    if args.make_z_positive:
        print(f"Applied global Z-positive fix: 180-degree rotation about {args.z_flip_axis.upper()} axis")
    print(f"serial_numbers = {serial_numbers}")
    print(f"frame_num      = {frame_num}")
    print("=" * 72)


if __name__ == "__main__":
    main()