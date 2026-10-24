#!/usr/bin/env python3
r"""
OAK-only AnyGrasp data collection wrapper (stage 01 of scripts/anygrasp_poses_one_shot.sh).

Usage:
# from the repository root
set -a; source config/aura.env; set +a
conda activate "$AURA_ENV"
python aura/capture/anygrasp_data_collection.py \
  --resolution high \
  --fps 15 \
  --warmup-seconds 4 \
  --record-duration 1 \
  --capture-bundle-index 0 \
  --sync-tolerance-ms 70 \
  --preview True \
  --output-root "$AURA_DATA_ROOT/debug/Anygrasp_capture_test" \
  --overwrite

Pipeline:
  1) record a short synchronized 3-OAK RGB-D sequence with DepthAI v3 recorder
  2) convert one synchronized bundle to PhysTwin/AnyGrasp folder layout
  3) generate calibrate.pkl + metadata.json from current OAK intrinsics/extrinsics

Expected layout (<repo> = --project-root, default $AURA_REPO_ROOT, else the checkout
containing this file; $AURA_CALIB_ROOT defaults to <repo>/calibration):
  <repo>/aura/capture/record_3oak.py
  <repo>/aura/conversion/raw_data_to_phystwin_style.py
  <repo>/aura/conversion/calibration_matrix_to_phystwin_style.py
  $AURA_CALIB_ROOT/intrinsic_results/640x400/oak_<MXID>.npz
  $AURA_CALIB_ROOT/intrinsic_results/1280x720/oak_<MXID>.npz
  $AURA_CALIB_ROOT/extrinsic_results/oak_<MXID>.npz
  (the precheck requires the .npz files; calibration_matrix_to_phystwin_style.py
  prefers a matching .json in the same folder when one exists)

Cameras: --oak-ids defaults to $AURA_OAK_IDS (space-separated MXIDs, camera-index
order); unless --camera-order is given, that order is camera index 0,1,2 of
color/, depth/ and calibrate.pkl. The child scripts run with this interpreter
(sys.executable), so start the wrapper from the AuRA conda environment.

Important:
  - This script uses camera intrinsics/extrinsics only.
  - It does NOT use robot_calib_result.npz/json. Robot-world calibration is for later
    grasp execution / world-to-base conversion, not for generating calibrate.pkl.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

DEFAULT_OAK_IDS = os.environ.get("AURA_OAK_IDS", "").split()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


def quote_cmd(cmd: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_cmd(cmd: Sequence[object], dry_run: bool = False) -> None:
    print("\n" + "=" * 88)
    print("Running command:")
    print(quote_cmd(cmd))
    print("=" * 88)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], check=True)


def first_existing(candidates: Sequence[Path], label: str) -> Path:
    for p in candidates:
        if p.is_file():
            return p.resolve()
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n  {tried}")


def first_existing_dir(candidates: Sequence[Path], label: str) -> Path:
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    tried = "\n  ".join(str(p) for p in candidates)
    raise NotADirectoryError(f"Could not find {label}. Tried:\n  {tried}")


def remove_if_overwrite(path: Path, overwrite: bool, label: str) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{label} already exists: {path}. Use --overwrite to replace it.")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def find_bundle_dirs(raw_root: Path) -> List[Path]:
    if not raw_root.is_dir():
        raise NotADirectoryError(f"Raw recording folder not found: {raw_root}")
    bundles = sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("bundle_"))
    if not bundles:
        raise FileNotFoundError(f"No bundle_* folders found under: {raw_root}")
    return bundles


def _repo_root() -> Path:
    """Repository root: $AURA_REPO_ROOT, else the directory two levels above aura/capture/."""
    return Path(os.environ.get("AURA_REPO_ROOT") or Path(__file__).resolve().parents[2])


def calib_root_from_project_root(project_root: Path) -> Path:
    """Calibration results root: $AURA_CALIB_ROOT, else <project-root>/calibration."""
    return Path(os.environ.get("AURA_CALIB_ROOT") or project_root / "calibration")


def default_intrinsic_dir(project_root: Path, resolution: str) -> Path:
    # Release layout:
    #   $AURA_CALIB_ROOT/intrinsic_results/{640x400,1280x720}
    # ($AURA_CALIB_ROOT defaults to <project-root>/calibration).
    calib_root = calib_root_from_project_root(project_root)
    if resolution == "low":
        return calib_root / "intrinsic_results" / "640x400"
    return calib_root / "intrinsic_results" / "1280x720"


def default_extrinsic_dir(project_root: Path) -> Path:
    calib_root = calib_root_from_project_root(project_root)
    return calib_root / "extrinsic_results"


def require_calibration_files(intrinsic_dir: Path, extrinsic_dir: Path, camera_order: Sequence[str]) -> None:
    missing = []
    for cam in camera_order:
        if not (intrinsic_dir / f"{cam}.npz").is_file():
            missing.append(str(intrinsic_dir / f"{cam}.npz"))
        if not (extrinsic_dir / f"{cam}.npz").is_file():
            missing.append(str(extrinsic_dir / f"{cam}.npz"))
    if missing:
        raise FileNotFoundError(
            "Missing required calibration files:\n  " + "\n  ".join(missing)
        )


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(
        description="Collect one AnyGrasp-ready 3-OAK RGB-D snapshot and convert it to the PhysTwin layout."
    )

    p.add_argument("--project-root", type=Path, default=_repo_root(),
                   help="Repository root; the default script paths and, when AURA_CALIB_ROOT is unset, the calibration root <project-root>/calibration are resolved from it. Default: $AURA_REPO_ROOT, else the repository containing this script.")
    p.add_argument("--record-script", type=Path, default=None,
                   help="Path to record_3oak.py. Default: <project-root>/aura/capture/record_3oak.py, else aura/capture/ of this checkout.")
    p.add_argument("--raw-convert-script", type=Path, default=None,
                   help="Path to raw_data_to_phystwin_style.py. Default: <project-root>/aura/conversion/raw_data_to_phystwin_style.py, else aura/conversion/ of this checkout.")
    p.add_argument("--calibration-script", type=Path, default=None,
                   help="Path to calibration_matrix_to_phystwin_style.py. Default: <project-root>/aura/conversion/calibration_matrix_to_phystwin_style.py, else aura/conversion/ of this checkout.")

    p.add_argument("--output-root", type=Path, default=None,
                   help="Final AnyGrasp-ready dataset folder.")
    p.add_argument("--dataset-name", type=str, default=f"Anygrasp_capture_3oak_{timestamp}")
    p.add_argument("--work-root", type=Path, default=None,
                   help="Temporary folder. Default: <output-root>_work")
    p.add_argument("--keep-work-root", action="store_true", help="Keep the work root: raw_recording/ (the raw bundles and recording_summary.json) and phystwin_3oak/ (intermediate conversion). By default it is deleted after a successful run.")
    p.add_argument("--overwrite", action="store_true", help="Delete an existing --output-root and --work-root before recording. Without it, an existing folder stops the run. --dry-run neither checks nor deletes.")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--oak-ids", nargs="+", default=list(DEFAULT_OAK_IDS),
                   help="Three OAK MXIDs to record. Defaults to $AURA_OAK_IDS, camera-index order.")
    p.add_argument("--camera-order", nargs="+", default=None,
                   help="Camera order for color/0,1,2 and depth/0,1,2. Default: oak_<MXID> in --oak-ids order.")
    p.add_argument("--serial-numbers", nargs="+", default=None,
                   help="Serial numbers written into metadata.json. Default: same as camera-order.")

    p.add_argument("--resolution", choices=("low", "high"), default="high",
                   help="Use low=640x400 or high=1280x720. Must match intrinsic-dir.")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--sensor-fps", type=int, default=15,
                   help="Sensor FPS passed to record_3oak.py. Keep 15 if 10 FPS gives black screen.")
    p.add_argument("--sync-fps", type=float, default=15.0,
                   help="Offline sync FPS passed to record_3oak.py.")
    p.add_argument("--warmup-seconds", type=float, default=2.5)
    p.add_argument("--record-duration", type=float, default=0.8,
                   help="Seconds to record. The wrapper converts one selected bundle. Default: 0.8.")
    p.add_argument("--capture-bundle-index", type=int, default=0,
                   help="Which sorted bundle to convert. Default: 0.")
    p.add_argument("--preview", type=str2bool, default=False, metavar="True|False")
    p.add_argument("--sync-tolerance-ms", type=float, default=70.0,
                   help="Offline sync tolerance. 70 ms was more forgiving for the 3-OAK setup.")
    p.add_argument("--oak-internal-sync-ms", type=float, default=25.0)
    p.add_argument("--enable-undistortion", type=str2bool, default=True, metavar="True|False")

    p.add_argument("--intrinsic-dir", type=Path, default=None,
                   help="Default: $AURA_CALIB_ROOT/intrinsic_results/1280x720 or 640x400 (per --resolution; $AURA_CALIB_ROOT defaults to <project-root>/calibration).")
    p.add_argument("--extrinsic-dir", type=Path, default=None,
                   help="Default: $AURA_CALIB_ROOT/extrinsic_results.")
    p.add_argument("--copy-meta", type=str2bool, default=True, metavar="True|False")
    p.add_argument("--force-single-wh", type=str2bool, default=True, metavar="True|False")
    p.add_argument("--make-z-positive", action="store_true",
                   help="Pass through to calibration_matrix_to_phystwin_style.py only if your downstream expects positive z. Not for the AuRA pipeline: robot_calib_result.npz expects the unrotated (z-down) world frame.")
    p.add_argument("--z-flip-axis", choices=("x", "y"), default="x")
    p.add_argument("--crop", nargs=5, action="append", default=[],
                   metavar=("CAMERA_NAME", "TOP", "BOTTOM", "LEFT", "RIGHT"),
                   help="Optional crop applied to both RGB/depth and calibration K. Usually not needed.")

    args = p.parse_args()
    if not args.oak_ids:
        p.error("no OAK camera ids: set AURA_OAK_IDS in config/aura.env and load it (set -a; source config/aura.env; set +a) or pass --oak-ids")
    return args


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    record_script = (args.record_script.resolve() if args.record_script else first_existing([
        project_root / "aura" / "capture" / "record_3oak.py",
        Path(__file__).resolve().parent / "record_3oak.py",
    ], "record_3oak.py"))

    raw_convert_script = (args.raw_convert_script.resolve() if args.raw_convert_script else first_existing([
        project_root / "aura" / "conversion" / "raw_data_to_phystwin_style.py",
        Path(__file__).resolve().parents[1] / "conversion" / "raw_data_to_phystwin_style.py",
    ], "raw_data_to_phystwin_style.py"))

    calibration_script = (args.calibration_script.resolve() if args.calibration_script else first_existing([
        project_root / "aura" / "conversion" / "calibration_matrix_to_phystwin_style.py",
        Path(__file__).resolve().parents[1] / "conversion" / "calibration_matrix_to_phystwin_style.py",
    ], "calibration_matrix_to_phystwin_style.py"))

    output_root = (args.output_root or Path.cwd() / args.dataset_name).resolve()
    work_root = (args.work_root or output_root.with_name(output_root.name + "_work")).resolve()
    raw_root = work_root / "raw_recording"
    intermediate_root = work_root / "phystwin_3oak"

    camera_order = args.camera_order or [f"oak_{mxid}" for mxid in args.oak_ids]
    if len(camera_order) != len(args.oak_ids):
        raise ValueError("--camera-order must have the same length as --oak-ids")
    serial_numbers = args.serial_numbers or camera_order
    if len(serial_numbers) != len(camera_order):
        raise ValueError("--serial-numbers must have the same length as --camera-order")

    intrinsic_dir = (args.intrinsic_dir.resolve() if args.intrinsic_dir else first_existing_dir([
        default_intrinsic_dir(project_root, args.resolution),
    ], "intrinsic dir (set AURA_CALIB_ROOT in config/aura.env or pass --intrinsic-dir)"))
    extrinsic_dir = (args.extrinsic_dir.resolve() if args.extrinsic_dir else first_existing_dir([
        default_extrinsic_dir(project_root),
    ], "extrinsic dir (set AURA_CALIB_ROOT in config/aura.env or pass --extrinsic-dir)"))

    require_calibration_files(intrinsic_dir, extrinsic_dir, camera_order)

    print("3-OAK AnyGrasp collection layout:")
    print(f"  project_root       = {project_root}")
    print(f"  record_script      = {record_script}")
    print(f"  raw_convert_script = {raw_convert_script}")
    print(f"  calibration_script = {calibration_script}")
    print(f"  intrinsic_dir      = {intrinsic_dir}")
    print(f"  extrinsic_dir      = {extrinsic_dir}")
    print(f"  output_root        = {output_root}")
    print(f"  work_root          = {work_root}")
    print(f"  oak_ids            = {args.oak_ids}")
    print(f"  camera_order       = {camera_order}")
    print("  note               = robot_calib_result is not used by this collection wrapper")

    if not args.dry_run:
        remove_if_overwrite(output_root, args.overwrite, "output root")
        remove_if_overwrite(work_root, args.overwrite, "work root")
        work_root.mkdir(parents=True, exist_ok=True)

    commands: List[List[object]] = []

    record_cmd: List[object] = [
        sys.executable,
        record_script,
        "--oak-ids",
        *args.oak_ids,
        "--resolution",
        args.resolution,
        "--fps",
        args.fps,
        "--sensor-fps",
        args.sensor_fps,
        "--sync-fps",
        args.sync_fps,
        "--output-root",
        raw_root,
        "--warmup-seconds",
        args.warmup_seconds,
        "--duration",
        args.record_duration,
        "--preview",
        str(bool(args.preview)),
        "--sync-tolerance-ms",
        args.sync_tolerance_ms,
        "--oak-internal-sync-ms",
        args.oak_internal_sync_ms,
        "--enable-undistortion",
        str(bool(args.enable_undistortion)),
        "--overwrite",
    ]
    commands.append(record_cmd)
    run_cmd(record_cmd, args.dry_run)

    if not args.dry_run:
        bundles = find_bundle_dirs(raw_root)
        if args.capture_bundle_index >= len(bundles):
            raise RuntimeError(
                f"--capture-bundle-index {args.capture_bundle_index} is out of range; "
                f"only {len(bundles)} bundles were captured."
            )
        print(
            f"Captured {len(bundles)} bundles. Converting bundle index "
            f"{args.capture_bundle_index}: {bundles[args.capture_bundle_index].name}"
        )

    convert_start = int(args.capture_bundle_index)
    convert_end = convert_start + 1
    convert_cmd: List[object] = [
        sys.executable,
        raw_convert_script,
        "--raw-root",
        raw_root,
        "--output-root",
        intermediate_root,
        "--camera-order",
        *camera_order,
        "--start-frame",
        convert_start,
        "--end-frame",
        convert_end,
        "--stride",
        1,
        "--overwrite",
    ]
    if args.copy_meta:
        convert_cmd.append("--copy-meta")
    for crop in args.crop:
        convert_cmd.extend(["--crop", *crop])
    commands.append(convert_cmd)
    run_cmd(convert_cmd, args.dry_run)

    # Preserve raw conversion metadata before calibration script overwrites metadata.json.
    if not args.dry_run:
        conv_meta = intermediate_root / "metadata.json"
        if conv_meta.exists():
            shutil.copy2(conv_meta, intermediate_root / "conversion_metadata.json")

    calib_cmd: List[object] = [
        sys.executable,
        calibration_script,
        "--intrinsic-dir",
        intrinsic_dir,
        "--extrinsic-dir",
        extrinsic_dir,
        "--output-dir",
        intermediate_root,
        "--camera-order",
        *camera_order,
        "--serial-numbers",
        *serial_numbers,
        "--frame-num",
        1,
        "--overwrite",
    ]
    if args.force_single_wh:
        calib_cmd.append("--force-single-wh")
    if args.make_z_positive:
        calib_cmd.extend(["--make-z-positive", "--z-flip-axis", args.z_flip_axis])
    for crop in args.crop:
        calib_cmd.extend(["--crop", *crop])
    commands.append(calib_cmd)
    run_cmd(calib_cmd, args.dry_run)

    if args.dry_run:
        print("\nDry run finished. No files were written.")
        return

    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(intermediate_root, output_root)

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "OAK-only AnyGrasp one-shot collection",
        "output_root": str(output_root),
        "work_root": str(work_root),
        "work_root_removed": not args.keep_work_root,
        "resolution": args.resolution,
        "fps": args.fps,
        "sensor_fps": args.sensor_fps,
        "sync_fps": args.sync_fps,
        "record_duration": args.record_duration,
        "capture_bundle_index": args.capture_bundle_index,
        "oak_ids": list(args.oak_ids),
        "camera_order": list(camera_order),
        "serial_numbers": list(serial_numbers),
        "intrinsic_dir": str(intrinsic_dir),
        "extrinsic_dir": str(extrinsic_dir),
        "robot_calib_used": False,
        "generated_layout": {
            "color": "color/{camera_index}/0.png",
            "depth": "depth/{camera_index}/0.npy",
            "calibration": "calibrate.pkl",
            "metadata": "metadata.json",
        },
        "commands": [quote_cmd(c) for c in commands],
    }
    with open(output_root / "anygrasp_collection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.keep_work_root:
        shutil.rmtree(work_root)

    print("\n" + "=" * 88)
    print("3-OAK AnyGrasp data collection finished.")
    print(f"Final folder: {output_root}")
    print("Expected layout:")
    print("  color/0/0.png, color/1/0.png, color/2/0.png")
    print("  depth/0/0.npy, depth/1/0.npy, depth/2/0.npy")
    print("  calibrate.pkl")
    print("  metadata.json")
    print("  anygrasp_collection_summary.json")
    print("=" * 88)


if __name__ == "__main__":
    main()
