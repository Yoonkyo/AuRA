#!/usr/bin/env python3
r"""Convert a synchronized raw recording into the PhysTwin-style case layout.

Input (--raw-root): a directory of bundle_NNNNNN/ folders, as written by
aura/capture/record_3oak.py, aura/robot/execute_anygrasp_franka_record_lift.py
and aura/robot/gamepad_record_franka_3oak.py:
    bundle_NNNNNN/oak_<MXID>/rgb.jpg     colour image (JPEG)
    bundle_NNNNNN/oak_<MXID>/depth.npy   HxW depth (uint16, millimetres)
    bundle_NNNNNN/meta.json              per-bundle recorder metadata

Output (--output-root):
    color/<camera_index>/<frame_index>.png   PNG re-encode of rgb.jpg
    depth/<camera_index>/<frame_index>.npy   depth.npy, values unchanged
    source_meta/<frame_index>.json           copy of meta.json (--copy-meta)
    metadata.json                            conversion record (camera order,
                                             frame selection, per-frame paths)
    color/<camera_index>.mp4                 only with --make-mp4
--crop, if given, is applied to both rgb and depth.

Pitfalls:
  - camera_index follows --camera-order; without it the camera folders of the
    first selected bundle are used in alphabetical order. Nothing checks this
    against calibrate.pkl, so pass the same --camera-order (oak_<MXID> for each
    id in $AURA_OAK_IDS, camera-index order) here and to
    calibration_matrix_to_phystwin_style.py.
  - --start-frame/--end-frame/--stride slice the sorted bundle list
    (--end-frame is exclusive); output frame indices always restart at 0.
  - Without --overwrite, existing color/, depth/ and source_meta/ folders
    are reused and files are replaced index by index: frames left over from
    an earlier, longer conversion stay on disk although metadata.json lists
    only the new ones, so a frame count taken from color/0/*.png is too
    high. Pass --overwrite when re-converting (all AuRA pipelines do).
  - calibration_matrix_to_phystwin_style.py, run next on the same directory,
    overwrites metadata.json: copy it to conversion_metadata.json first (all
    AuRA pipelines do).
  - --make-mp4 uses the 'avc1' (H.264) fourcc, which many pip/conda OpenCV
    builds cannot open; the script then raises "Failed to open video
    writer" after all frames are written and before metadata.json is
    saved. No AuRA pipeline passes it: scripts/run_anygrasp_to_lift_one_shot.sh
    (stage 03b) and scripts/collect_hand_phystwin_one_shot.sh (stage 05)
    write color/<i>.mp4 themselves (mp4v/FMP4/MJPG). --fps defaults to 30
    while the AuRA recorders default to 15 fps: pass the capture rate.
  - metadata.json records the input and output paths as passed on the
    command line.

Example
-------
# from the repository root (the arguments of stage 03 of
# scripts/run_anygrasp_to_lift_one_shot.sh)
set -a; source config/aura.env; set +a
conda activate "$AURA_ENV"
CASE_DIR="$AURA_DATA_ROOT/franka_lift/<case>/phystwin"
python aura/conversion/raw_data_to_phystwin_style.py \
  --raw-root "$CASE_DIR/raw_recording" \
  --output-root "$CASE_DIR" \
  --camera-order $(printf 'oak_%s ' $AURA_OAK_IDS) \
  --start-frame 0 --stride 1 --copy-meta --overwrite &&
# calibration_matrix_to_phystwin_style.py replaces metadata.json: keep a copy
cp "$CASE_DIR/metadata.json" "$CASE_DIR/conversion_metadata.json"
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CropSpec:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
        }


def parse_crop_arg(values: Sequence[str]) -> Tuple[str, CropSpec]:
    if len(values) != 5:
        raise argparse.ArgumentTypeError(
            "--crop expects 5 values: CAMERA_NAME TOP BOTTOM LEFT RIGHT"
        )
    cam = values[0]
    try:
        top, bottom, left, right = (int(v) for v in values[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Crop values must be integers") from exc

    if min(top, bottom, left, right) < 0:
        raise argparse.ArgumentTypeError("Crop values must be >= 0")
    return cam, CropSpec(top=top, bottom=bottom, left=left, right=right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert synchronized raw bundles into PhysTwin-style color/depth folders."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Directory containing bundle_NNNNNN/ folders, e.g. data/franka_lift/<case>/phystwin/raw_recording",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output dataset root, e.g. data/franka_lift/<case>/phystwin",
    )
    parser.add_argument(
        "--camera-order",
        nargs="+",
        default=None,
        help=(
            "Optional explicit camera order. Example: "
            "--camera-order oak_<MXID_0> oak_<MXID_1> oak_<MXID_2> "
            "(the camera folder names in each bundle, in camera-index order = the order of "
            "$AURA_OAK_IDS; default: alphabetical folder order)"
        ),
    )
    parser.add_argument(
        "--rgb-name",
        type=str,
        default="rgb.jpg",
        help="RGB filename inside each camera folder (default: rgb.jpg)",
    )
    parser.add_argument(
        "--depth-name",
        type=str,
        default="depth.npy",
        help="Depth filename inside each camera folder (default: depth.npy)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help=(
            "Start bundle index after sorting. For standard names like bundle_000074, "
            "--start-frame 74 starts from that bundle."
        ),
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="End bundle index (exclusive) after sorting; default uses all",
    )
    parser.add_argument("--stride", type=int, default=1, help="Take every N-th bundle")
    parser.add_argument(
        "--crop",
        nargs=5,
        action="append",
        metavar=("CAMERA_NAME", "TOP", "BOTTOM", "LEFT", "RIGHT"),
        default=[],
        help=(
            "Optional per-camera crop applied to BOTH rgb and depth. "
            "Example: --crop oak_<MXID_0> 40 40 0 0"
        ),
    )
    parser.add_argument(
        "--copy-meta",
        action="store_true",
        help="Copy each bundle's meta.json into output_root/source_meta/{frame_idx}.json",
    )
    parser.add_argument(
        "--make-mp4",
        action="store_true",
        help=(
            "Also write color/0.mp4, color/1.mp4, ... from converted PNG frames. Do not use it in "
            "the AuRA env: its pinned OpenCV cannot write the avc1 fourcc, so the run fails with "
            "the frames written but no metadata.json."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS to use when creating mp4 previews (default: 30; the AuRA recorders default to 15: pass the capture rate)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing color/depth/source_meta folders under output-root before writing",
    )
    return parser.parse_args()


def find_bundles(raw_root: Path) -> List[Path]:
    bundles = [p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("bundle_")]
    bundles.sort(key=lambda p: p.name)
    if not bundles:
        raise FileNotFoundError(f"No bundle_* folders found under {raw_root}")
    return bundles


def discover_cameras(bundle_dir: Path) -> List[str]:
    cameras = [p.name for p in bundle_dir.iterdir() if p.is_dir()]
    cameras.sort()
    if not cameras:
        raise RuntimeError(f"No camera folders found inside {bundle_dir}")
    return cameras


def prepare_output_dirs(output_root: Path, num_cameras: int, copy_meta: bool, overwrite: bool) -> None:
    color_root = output_root / "color"
    depth_root = output_root / "depth"
    meta_root = output_root / "source_meta"

    if overwrite:
        for p in [color_root, depth_root, meta_root]:
            if p.exists():
                shutil.rmtree(p)

    color_root.mkdir(parents=True, exist_ok=True)
    depth_root.mkdir(parents=True, exist_ok=True)
    if copy_meta:
        meta_root.mkdir(parents=True, exist_ok=True)

    for cam_idx in range(num_cameras):
        (color_root / str(cam_idx)).mkdir(parents=True, exist_ok=True)
        (depth_root / str(cam_idx)).mkdir(parents=True, exist_ok=True)


def apply_crop(arr: np.ndarray, crop: CropSpec, src_path: Path) -> np.ndarray:
    h, w = arr.shape[:2]
    y0 = crop.top
    y1 = h - crop.bottom
    x0 = crop.left
    x1 = w - crop.right

    if y0 >= y1 or x0 >= x1:
        raise RuntimeError(
            f"Invalid crop for {src_path}: input {w}x{h}, crop={crop.as_dict()} gives empty output"
        )
    return arr[y0:y1, x0:x1].copy()


def load_rgb_and_size(image_path: Path) -> Tuple[np.ndarray, List[int]]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    return img, [int(w), int(h)]


def parse_crop_specs(args: argparse.Namespace) -> Dict[str, CropSpec]:
    crop_map: Dict[str, CropSpec] = {}
    for raw in args.crop:
        cam, crop = parse_crop_arg(raw)
        crop_map[cam] = crop
    return crop_map


def convert_frame(
    bundle_dir: Path,
    frame_idx: int,
    cameras: Sequence[str],
    args: argparse.Namespace,
    crop_map: Dict[str, CropSpec],
) -> Dict:
    frame_info = {
        "frame_idx": frame_idx,
        "bundle_name": bundle_dir.name,
        "bundle_path": str(bundle_dir),
        "cameras": {},
    }

    for cam_idx, cam_name in enumerate(cameras):
        cam_dir = bundle_dir / cam_name
        rgb_in = cam_dir / args.rgb_name
        depth_in = cam_dir / args.depth_name

        if not rgb_in.exists():
            raise FileNotFoundError(f"Missing RGB file: {rgb_in}")
        if not depth_in.exists():
            raise FileNotFoundError(f"Missing depth file: {depth_in}")

        rgb_out = args.output_root / "color" / str(cam_idx) / f"{frame_idx}.png"
        depth_out = args.output_root / "depth" / str(cam_idx) / f"{frame_idx}.npy"

        rgb, rgb_src_size = load_rgb_and_size(rgb_in)
        depth = np.load(depth_in)
        if depth.ndim != 2:
            raise RuntimeError(f"Expected depth to be HxW in {depth_in}, got shape {depth.shape}")

        crop = crop_map.get(cam_name, CropSpec())
        if any(v != 0 for v in (crop.top, crop.bottom, crop.left, crop.right)):
            rgb = apply_crop(rgb, crop, rgb_in)
            depth = apply_crop(depth, crop, depth_in)

        rgb_h, rgb_w = rgb.shape[:2]
        depth_h, depth_w = depth.shape[:2]
        if (rgb_h, rgb_w) != (depth_h, depth_w):
            raise RuntimeError(
                f"RGB/depth size mismatch after crop for {cam_name} in {bundle_dir.name}: "
                f"rgb={rgb_w}x{rgb_h}, depth={depth_w}x{depth_h}"
            )

        ok = cv2.imwrite(str(rgb_out), rgb)
        if not ok:
            raise RuntimeError(f"Failed to write PNG image: {rgb_out}")
        np.save(depth_out, depth)

        frame_info["cameras"][cam_name] = {
            "camera_index": cam_idx,
            "rgb_src": str(rgb_in),
            "depth_src": str(depth_in),
            "rgb_dst": str(rgb_out),
            "depth_dst": str(depth_out),
            "rgb_src_size": rgb_src_size,
            "depth_src_size": [int(depth.shape[1] + crop.left + crop.right), int(depth.shape[0] + crop.top + crop.bottom)]
            if any(v != 0 for v in (crop.top, crop.bottom, crop.left, crop.right))
            else [int(depth.shape[1]), int(depth.shape[0])],
            "output_size": [int(rgb_w), int(rgb_h)],
            "crop": crop.as_dict(),
        }

    if args.copy_meta:
        src_meta = bundle_dir / "meta.json"
        if src_meta.exists():
            dst_meta = args.output_root / "source_meta" / f"{frame_idx}.json"
            shutil.copy2(src_meta, dst_meta)
            frame_info["meta_src"] = str(src_meta)
            frame_info["meta_dst"] = str(dst_meta)

    return frame_info


def make_mp4_for_camera(color_dir: Path, mp4_path: Path, fps: float) -> None:
    frame_paths = sorted(color_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not frame_paths:
        return

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read first PNG for video: {frame_paths[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {mp4_path}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame for mp4: {frame_path}")
            if frame.shape[0] != h or frame.shape[1] != w:
                raise RuntimeError(
                    f"Inconsistent frame size in {color_dir}: {frame_path} has {frame.shape[1]}x{frame.shape[0]}, expected {w}x{h}"
                )
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root
    output_root = args.output_root

    if not raw_root.exists():
        raise FileNotFoundError(f"raw-root does not exist: {raw_root}")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if args.end_frame is not None and args.end_frame < 0:
        raise ValueError("--end-frame must be >= 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

    bundles = find_bundles(raw_root)
    selected = bundles[args.start_frame:args.end_frame:args.stride]
    if not selected:
        raise RuntimeError("No bundles selected after applying start/end/stride")

    auto_cameras = discover_cameras(selected[0])
    cameras = args.camera_order if args.camera_order is not None else auto_cameras

    missing = [cam for cam in cameras if cam not in auto_cameras]
    if missing:
        raise RuntimeError(
            f"These cameras are not present in the first selected bundle {selected[0].name}: {missing}. "
            f"Available cameras: {auto_cameras}"
        )

    crop_map = parse_crop_specs(args)
    unknown_crop_cams = [cam for cam in crop_map if cam not in cameras]
    if unknown_crop_cams:
        raise RuntimeError(
            f"Crop specified for cameras not in camera-order/selected cameras: {unknown_crop_cams}"
        )

    prepare_output_dirs(output_root, len(cameras), args.copy_meta, args.overwrite)

    input_sizes = {}
    output_sizes = {}
    for cam_name in cameras:
        first_rgb = selected[0] / cam_name / args.rgb_name
        first_img, input_sizes[cam_name] = load_rgb_and_size(first_rgb)
        crop = crop_map.get(cam_name, CropSpec())
        first_img = apply_crop(first_img, crop, first_rgb) if any(crop.as_dict().values()) else first_img
        h, w = first_img.shape[:2]
        output_sizes[cam_name] = [int(w), int(h)]

    frame_records = []
    for frame_idx, bundle_dir in enumerate(selected):
        info = convert_frame(bundle_dir, frame_idx, cameras, args, crop_map)
        frame_records.append(info)
        print(f"[{frame_idx:04d}] converted {bundle_dir.name}")

    if args.make_mp4:
        for cam_idx in range(len(cameras)):
            color_dir = output_root / "color" / str(cam_idx)
            mp4_path = output_root / "color" / f"{cam_idx}.mp4"
            make_mp4_for_camera(color_dir, mp4_path, args.fps)
            print(f"Saved mp4: {mp4_path}")

    metadata = {
        "dataset_name": output_root.name,
        "source_recording": str(raw_root),
        "camera_order": cameras,
        "camera_index_map": {cam: idx for idx, cam in enumerate(cameras)},
        "num_cameras": len(cameras),
        "num_frames": len(selected),
        "frame_selection": {
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "stride": args.stride,
            "selected_bundle_names": [p.name for p in selected],
        },
        "rgb_filename": args.rgb_name,
        "depth_filename": args.depth_name,
        "input_image_sizes": input_sizes,
        "output_image_sizes": output_sizes,
        "crop_by_camera": {cam: crop.as_dict() for cam, crop in crop_map.items()},
        "generated_layout": {
            "color": "color/{camera_index}/{frame_index}.png",
            "depth": "depth/{camera_index}/{frame_index}.npy",
        },
        "frames": frame_records,
    }

    metadata_path = output_root / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")
    print("Done.")


if __name__ == "__main__":
    main()
