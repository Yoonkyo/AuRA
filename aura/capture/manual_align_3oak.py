#!/usr/bin/env python3
r"""Manual scene alignment preview: live OAK RGB fused with a reference overlay.

Standalone and camera-only: no robot, no gripper, no gamepad, no recording,
no fr3/pylibfranka imports. Runs in the AuRA conda environment ($AURA_ENV in
config/aura.env).

Uses:
  - Aim the cameras and match MXIDs to physical cameras (docs/hardware.md).
  - Re-stage a scene from an earlier case: align the scene/objects to the
    overlay, quit (q/ESC), then run the preparation script (docs/usage.md).

Overlay sources (mutually exclusive; with neither, a plain live preview):
  --overlay-reference-root  a v1 recording folder, not a robot case folder:
                            raw (bundle_XXXXXX/oak_<MXID>/rgb.jpg) or converted
                            (color/<cam>/<frame>.png, camera order from
                            metadata.json camera_names), e.g. <case>/anygrasp
                            (static snapshot, to re-stage), <case>/phystwin
                            (frame 0: lift onset) or a human-baseline case.
  --overlay-mask-pkl        needs <case>/mask/processed_masks.pkl, which is
                            written by the v2 post-processing stage; v1 does
                            not produce it.

Cameras: --oak-ids defaults to $AURA_OAK_IDS (camera-index order); that order
also maps the processed_masks.pkl camera indices to cameras.

The RGB outputs use the same DepthAI v3 request (size, BGR888p, undistortion)
as the recording scripts, so the preview geometry matches the recordings.

Example
-------
# from the repository root
set -a; source config/aura.env; set +a
conda activate "$AURA_ENV"
python aura/capture/manual_align_3oak.py \
  --resolution high --fps 15 --sensor-fps 15 \
  --overlay-reference-root "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp" \
  --overlay-reference-frame 0

Controls: q or ESC in the preview window quits.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import depthai as dai
except Exception as e:
    raise RuntimeError("Failed to import depthai. Activate the AuRA conda environment ($AURA_ENV, DepthAI v3) first.") from e


DEFAULT_OAK_IDS = os.environ.get("AURA_OAK_IDS", "").split()

RESOLUTION_PRESETS = {
    "low": {"rgb": (640, 400)},
    "high": {"rgb": (1280, 720)},
}

DEFAULT_FPS = 15
DEFAULT_SENSOR_FPS = 15
OAK_QUEUE_MAXSIZE = 4
PREVIEW_SCALE = 0.35


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


def mapping_get(mapping, key):
    if not isinstance(mapping, dict):
        raise TypeError(f"Expected mapping, got {type(mapping).__name__}")
    if key in mapping:
        return mapping[key]
    string_key = str(key)
    if string_key in mapping:
        return mapping[string_key]
    raise KeyError(key)


def load_preview_overlay_masks(
    path: Optional[Path],
    frame_index: int,
    oak_ids: List[str],
) -> Dict[str, np.ndarray]:
    """Load processed_masks[frame][cam]['object'] and map cam indices to OAK IDs."""
    if path is None:
        return {}
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        processed_masks = pickle.load(handle)
    frame_masks = mapping_get(processed_masks, int(frame_index))
    overlays: Dict[str, np.ndarray] = {}
    for camera_index, mxid in enumerate(oak_ids):
        try:
            camera_item = mapping_get(frame_masks, camera_index)
        except KeyError:
            continue
        if not isinstance(camera_item, dict) or "object" not in camera_item:
            continue
        mask = np.asarray(camera_item["object"], dtype=bool)
        if mask.ndim != 2 or not np.any(mask):
            continue
        overlays[f"oak_{mxid}"] = mask
    if not overlays:
        raise RuntimeError(
            f"No non-empty object masks for frame {frame_index} and cameras 0..{len(oak_ids)-1} in {path}"
        )
    return overlays


def _find_reference_rgb_path(
    root: Path,
    *,
    frame_index: int,
    camera_index: int,
    mxid: str,
) -> Optional[Path]:
    """Resolve one source RGB from either a raw recording or converted AuRA case."""
    bundle = root / f"bundle_{frame_index:06d}"
    camera_dir = root / "color" / str(camera_index)
    candidates = [
        bundle / f"oak_{mxid}" / "rgb.jpg",
        bundle / f"oak_{mxid}" / "rgb.png",
        bundle / mxid / "rgb.jpg",
        bundle / mxid / "rgb.png",
        camera_dir / f"{frame_index}.png",
        camera_dir / f"{frame_index:05d}.png",
        camera_dir / f"{frame_index:06d}.png",
        camera_dir / f"{frame_index}.jpg",
        camera_dir / f"{frame_index:05d}.jpg",
        camera_dir / f"{frame_index:06d}.jpg",
    ]
    return next((path for path in candidates if path.is_file()), None)


def load_preview_reference_frames(
    root: Optional[Path],
    frame_index: int,
    oak_ids: List[str],
) -> Dict[str, np.ndarray]:
    """Load one RGB per camera for full-frame alpha blending in live preview."""
    if root is None:
        return {}
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    camera_index_by_name: Dict[str, int] = {}
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            names = metadata.get("camera_names") or metadata.get("cameras") or []
            camera_index_by_name = {
                str(name): int(index) for index, name in enumerate(names)
            }
        except Exception as error:
            raise RuntimeError(f"Failed to parse camera order from {metadata_path}: {error}") from error

    overlays: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for fallback_camera_index, mxid in enumerate(oak_ids):
        camera_index = camera_index_by_name.get(
            f"oak_{mxid}",
            camera_index_by_name.get(mxid, fallback_camera_index),
        )
        path = _find_reference_rgb_path(
            root,
            frame_index=frame_index,
            camera_index=camera_index,
            mxid=mxid,
        )
        if path is None:
            missing.append(f"cam{camera_index}:{mxid}")
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read reference RGB: {path}")
        overlays[f"oak_{mxid}"] = image

    if missing:
        raise FileNotFoundError(
            f"Missing reference RGB frame {frame_index} for {missing} below {root}. "
            "Expected raw bundle_XXXXXX/oak_<MXID>/rgb.jpg or converted color/<cam>/<frame>.png."
        )
    return overlays


def maybe_resize(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def open_oak_rgb(
    exit_stack: contextlib.ExitStack,
    mxid: str,
    output_fps: int,
    sensor_fps: int,
    rgb_size: Tuple[int, int],
    enable_undistortion: bool,
):
    print(f"Opening OAK by MXID: {mxid}", flush=True)
    device = dai.Device(dai.DeviceInfo(mxid))
    pipeline = exit_stack.enter_context(dai.Pipeline(device))
    opened_mxid = pipeline.getDefaultDevice().getDeviceInfo().getDeviceId()
    if opened_mxid != mxid:
        raise RuntimeError(f"Requested {mxid}, but pipeline opened {opened_mxid}")

    with contextlib.suppress(Exception):
        print(f"[{mxid}] USB speed={device.getUsbSpeed()}", flush=True)

    cam_rgb = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A,
        sensorFps=sensor_fps,
    )
    rgb_out = cam_rgb.requestOutput(
        size=rgb_size,
        type=dai.ImgFrame.Type.BGR888p,
        fps=output_fps,
        enableUndistortion=enable_undistortion,
    )
    try:
        q = rgb_out.createOutputQueue(maxSize=OAK_QUEUE_MAXSIZE, blocking=False)
    except TypeError:
        q = rgb_out.createOutputQueue()
    pipeline.start()
    print(f"Opened oak_{mxid}", flush=True)
    return q


def render_preview(
    latest: Dict[str, Optional[np.ndarray]],
    scale: float,
    overlay_masks: Dict[str, np.ndarray],
    reference_frames: Dict[str, np.ndarray],
    overlay_alpha: float,
) -> bool:
    """Draw fused panels; return False when the user quits (q/ESC)."""
    panels = []
    for source_name, rgb_frame in sorted(latest.items()):
        if rgb_frame is None:
            continue
        rgb_full = rgb_frame.copy()
        reference = reference_frames.get(source_name)
        if reference is not None:
            if reference.shape[:2] != rgb_full.shape[:2]:
                reference = cv2.resize(
                    reference,
                    (rgb_full.shape[1], rgb_full.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            rgb_full = cv2.addWeighted(
                rgb_full,
                1.0 - float(overlay_alpha),
                reference,
                float(overlay_alpha),
                0.0,
            )
        mask = overlay_masks.get(source_name)
        if mask is not None:
            if mask.shape != rgb_full.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (rgb_full.shape[1], rgb_full.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            tint = np.zeros_like(rgb_full)
            tint[..., 1] = 255
            blended = (
                (1.0 - float(overlay_alpha)) * rgb_full[mask].astype(np.float32)
                + float(overlay_alpha) * tint[mask].astype(np.float32)
            )
            rgb_full[mask] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(rgb_full, contours, -1, (0, 0, 255), 2)
        rgb = maybe_resize(rgb_full, scale)
        cv2.putText(rgb, source_name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        panels.append(rgb)
    if panels:
        view = np.hstack(panels)
        cv2.imshow("OAK alignment preview", view)
        key = cv2.waitKey(1)
        if key in (ord("q"), 27):
            return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camera-only manual alignment preview (fused OAK image), no robot/gamepad."
    )
    p.add_argument(
        "--oak-ids",
        nargs="+",
        default=DEFAULT_OAK_IDS,
        help="OAK MXIDs; defaults to $AURA_OAK_IDS, camera-index order.",
    )
    p.add_argument(
        "--resolution",
        choices=sorted(RESOLUTION_PRESETS.keys()),
        default="high",
        help="Use the same value you will pass to the recording script.",
    )
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--sensor-fps", type=int, default=DEFAULT_SENSOR_FPS)
    p.add_argument("--enable-undistortion", type=str2bool, default=True)
    p.add_argument("--preview-scale", type=float, default=PREVIEW_SCALE)
    p.add_argument(
        "--overlay-mask-pkl",
        type=Path,
        default=None,
        help=(
            "Optional source case mask/processed_masks.pkl (written by the v2 post-processing "
            "stage; v1 does not produce it). The selected frame's object masks "
            "are overlaid on live OAK previews using --oak-ids order as camera indices."
        ),
    )
    p.add_argument("--overlay-frame", type=int, default=0)
    p.add_argument(
        "--overlay-reference-root",
        type=Path,
        default=None,
        help=(
            "Optional source recording folder, e.g. <case>/anygrasp (not the robot case folder itself). "
            "Alpha-blend one complete reference RGB per camera into the live preview. Supports "
            "bundle_XXXXXX/oak_<MXID>/rgb.jpg and converted color/<cam>/<frame>.png layouts."
        ),
    )
    p.add_argument("--overlay-reference-frame", type=int, default=0)
    p.add_argument("--overlay-alpha", type=float, default=0.28)
    args = p.parse_args()

    if not args.oak_ids:
        p.error("no OAK camera ids: set AURA_OAK_IDS in config/aura.env and load it "
                "(set -a; source config/aura.env; set +a) or pass --oak-ids")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be in [0,1]")
    if args.overlay_frame < 0:
        raise ValueError("--overlay-frame must be non-negative")
    if args.overlay_reference_frame < 0:
        raise ValueError("--overlay-reference-frame must be non-negative")
    if args.overlay_mask_pkl is not None and args.overlay_reference_root is not None:
        raise ValueError("Use either --overlay-mask-pkl or --overlay-reference-root, not both.")
    return args


def main() -> None:
    args = parse_args()

    overlay_masks = load_preview_overlay_masks(
        args.overlay_mask_pkl,
        frame_index=args.overlay_frame,
        oak_ids=args.oak_ids,
    )
    reference_frames = load_preview_reference_frames(
        args.overlay_reference_root,
        frame_index=args.overlay_reference_frame,
        oak_ids=args.oak_ids,
    )
    if overlay_masks:
        print(
            f"Overlay masks: {args.overlay_mask_pkl} frame={args.overlay_frame} "
            f"cameras={sorted(overlay_masks)}",
            flush=True,
        )
    if reference_frames:
        print(
            f"Reference RGB: {args.overlay_reference_root} "
            f"frame={args.overlay_reference_frame} cameras={sorted(reference_frames)}",
            flush=True,
        )
    if not overlay_masks and not reference_frames:
        print(
            "NOTE: no --overlay-mask-pkl or --overlay-reference-root given; "
            "showing plain live previews.",
            flush=True,
        )

    available = {d.getDeviceId() for d in dai.Device.getAllAvailableDevices()}
    missing = [mxid for mxid in args.oak_ids if mxid not in available]
    if missing:
        raise RuntimeError(f"Requested OAK(s) not detected: {missing}")

    rgb_size = tuple(RESOLUTION_PRESETS[args.resolution]["rgb"])
    latest: Dict[str, Optional[np.ndarray]] = {f"oak_{mxid}": None for mxid in args.oak_ids}

    with contextlib.ExitStack() as exit_stack:
        queues = {}
        for mxid in args.oak_ids:
            queues[f"oak_{mxid}"] = open_oak_rgb(
                exit_stack=exit_stack,
                mxid=mxid,
                output_fps=args.fps,
                sensor_fps=args.sensor_fps,
                rgb_size=rgb_size,
                enable_undistortion=args.enable_undistortion,
            )

        print("Alignment preview running. Press q or ESC in the window to quit.", flush=True)
        try:
            while True:
                got_any = False
                for source_name, q in queues.items():
                    with contextlib.suppress(Exception):
                        msg = q.tryGet()
                        while msg is not None:
                            latest[source_name] = msg.getCvFrame()
                            got_any = True
                            msg = q.tryGet()
                if not render_preview(
                    latest,
                    args.preview_scale,
                    overlay_masks,
                    reference_frames,
                    args.overlay_alpha,
                ):
                    break
                if not got_any:
                    time.sleep(0.005)
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    print("Alignment preview closed.", flush=True)


if __name__ == "__main__":
    main()
