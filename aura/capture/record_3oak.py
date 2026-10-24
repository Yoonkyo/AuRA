#!/usr/bin/env python3
"""
DepthAI v3 3-OAK RGB-D memory-first recorder with offline synchronization.

Purpose
-------
For Franka lift / PhysTwin dynamic data collection, online synchronization can drop
many timesteps because one camera frame may arrive late at the Python process.
This recorder avoids making online drop decisions:

  1. Warm up cameras and estimate each camera's receive-latency offset.
  2. Freeze those offsets.
  3. Record every RGB-D frame from every OAK into CPU RAM for at most 30 seconds.
  4. After recording finishes, run offline fixed-rate synchronization over all frames.
  5. Save synchronized bundles to disk.

This is a DepthAI v3-style script:
  - dai.Device(dai.DeviceInfo(mxid))
  - dai.Pipeline(device)
  - dai.node.Camera
  - cam.requestOutput(...)
  - output.createOutputQueue(...)
  - no XLinkOut / createColorCamera

Usage
-----
Run from the repository root in the AuRA conda environment ($AURA_ENV), with
config/aura.env loaded into the shell (set -a; source config/aura.env; set +a):

  python aura/capture/record_3oak.py --output-root <OUTPUT_DIR>
  python aura/capture/record_3oak.py --output-root <OUTPUT_DIR> --oak-ids <MXID> <MXID> <MXID>

Stage 01 (aura/capture/anygrasp_data_collection.py) and the human-baseline collector
(scripts/collect_hand_phystwin_one_shot.sh) run this recorder themselves; run it
directly for calibration captures.

Output layout
-------------
output_root/
  bundle_000000/
    oak_<MXID>/rgb.jpg
    oak_<MXID>/depth.npy
    meta.json
  bundle_000001/
    ...
  recording_summary.json

Notes
-----
- --oak-ids defaults to $AURA_OAK_IDS (space-separated MXIDs, camera-index order).
- This script stores raw RGB-D frames in CPU RAM. It hard-caps --duration to 30 s.
- RTX/VRAM is not used; this is CPU RAM memory-first recording.
- For Phystwin dynamic data, prefer sync_tolerance_ms around 30-45 ms.
  Static captures use more: 70 ms for the AnyGrasp snapshot, 75 ms for calibration.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import depthai as dai
except Exception as e:
    raise RuntimeError("Failed to import depthai. Activate the AuRA conda environment ($AURA_ENV, DepthAI v3) first.") from e


# OAK MXIDs in camera-index order, from AURA_OAK_IDS (config/aura.env).
DEFAULT_OAK_IDS = os.environ.get("AURA_OAK_IDS", "").split()

RESOLUTION_PRESETS = {
    "low": {"rgb": (640, 400), "mono": (640, 400)},
    "high": {"rgb": (1280, 720), "mono": (1280, 720)},
}

DEFAULT_DURATION_S = 30.0
MAX_DURATION_S = 30.0
DEFAULT_FPS = 15
DEFAULT_SENSOR_FPS = 15
DEFAULT_SYNC_FPS = 15.0
DEFAULT_SYNC_TOLERANCE_MS = 45.0
DEFAULT_INTERNAL_SYNC_MS = 25.0
DEFAULT_OFFSET_FREEZE_PERCENTILE = 5.0
OAK_QUEUE_MAXSIZE = 16
RGB_JPEG_QUALITY = 90
PREVIEW_SCALE = 0.35


# -------------------------
# Small helpers
# -------------------------
def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def td_to_sec(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x.total_seconds())
    except Exception:
        return None


def maybe_resize(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))


def get_mem_available_gib() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


def estimate_ram_gib(num_cams: int, rgb_size: Tuple[int, int], fps: float, duration_s: float) -> float:
    w, h = rgb_size
    # rgb uint8 BGR: 3 bytes/pixel, depth uint16: 2 bytes/pixel.
    raw_bytes_per_frame_per_cam = w * h * (3 + 2)
    raw = raw_bytes_per_frame_per_cam * num_cams * fps * duration_s
    # Python object/list/copy overhead. This is a deliberately conservative estimate.
    return raw * 1.6 / (1024.0 ** 3)


@dataclass
class MemFrame:
    source: str
    host_ts: float
    device_ts: Optional[float]
    match_ts: float
    frame_number: Optional[int]
    rgb: np.ndarray
    depth: np.ndarray


class DeviceTimeProjector:
    """Add each camera's receive latency to its DepthAI capture timestamps (host clock)."""

    def __init__(self, source_name: str, ema_alpha: float = 0.02):
        self.source_name = source_name
        self.ema_alpha = ema_alpha
        self.offset_ema: Optional[float] = None
        self.offset_samples: List[float] = []
        self.fixed_offset: Optional[float] = None
        self.last_device_ts: Optional[float] = None
        self.num_project_calls = 0

    def project(self, host_ts: float, device_ts: Optional[float]) -> float:
        self.num_project_calls += 1
        if device_ts is None:
            return host_ts

        if self.last_device_ts is not None and device_ts < self.last_device_ts:
            if self.fixed_offset is None:
                self.offset_samples.clear()
                self.offset_ema = None
        self.last_device_ts = device_ts

        observed_offset = host_ts - device_ts

        if self.fixed_offset is not None:
            return device_ts + self.fixed_offset

        self.offset_samples.append(observed_offset)
        if self.offset_ema is None:
            self.offset_ema = observed_offset
        else:
            a = self.ema_alpha
            self.offset_ema = (1.0 - a) * self.offset_ema + a * observed_offset
        return device_ts + self.offset_ema

    def freeze(self, percentile: float = DEFAULT_OFFSET_FREEZE_PERCENTILE) -> None:
        if self.fixed_offset is not None:
            return
        if self.offset_samples:
            self.fixed_offset = float(np.percentile(np.asarray(self.offset_samples), percentile))
        elif self.offset_ema is not None:
            self.fixed_offset = float(self.offset_ema)
        else:
            self.fixed_offset = 0.0

    def as_meta(self) -> Dict:
        vals = np.asarray(self.offset_samples, dtype=np.float64) if self.offset_samples else np.asarray([])
        meta = {
            "source_name": self.source_name,
            "fixed_offset": self.fixed_offset,
            "offset_ema": self.offset_ema,
            "num_offset_samples": int(len(self.offset_samples)),
            "num_project_calls": int(self.num_project_calls),
        }
        if vals.size:
            meta.update(
                {
                    "offset_samples_p05": float(np.percentile(vals, 5)),
                    "offset_samples_p50": float(np.percentile(vals, 50)),
                    "offset_samples_p95": float(np.percentile(vals, 95)),
                    "offset_samples_min": float(vals.min()),
                    "offset_samples_max": float(vals.max()),
                }
            )
        return meta


# -------------------------
# DepthAI v3 setup
# -------------------------
def list_available_oaks() -> List[str]:
    devices = dai.Device.getAllAvailableDevices()
    ids = []
    print("Available OAK devices:")
    for d in devices:
        mxid = d.getDeviceId()
        ids.append(mxid)
        print(f"  {mxid}  state={getattr(d, 'state', 'unknown')}")
    return ids


def build_oak_pipeline_v3(
    pipeline: dai.Pipeline,
    output_fps: int,
    sensor_fps: int,
    rgb_size: Tuple[int, int],
    mono_size: Tuple[int, int],
    oak_internal_sync_ms: float,
    enable_undistortion: bool,
):
    device = pipeline.getDefaultDevice()
    platform = device.getPlatform()
    mxid = device.getDeviceInfo().getDeviceId()

    print(f"[{mxid}] platform={platform}")
    try:
        print(f"[{mxid}] USB speed={device.getUsbSpeed()}")
    except Exception:
        pass
    print(
        f"[{mxid}] RGB size={rgb_size}, mono/depth size={mono_size}, "
        f"sensor_fps={sensor_fps}, output_fps={output_fps}, undistort={enable_undistortion}"
    )

    cam_rgb = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_A,
        sensorFps=sensor_fps,
    )
    cam_left = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B,
        sensorFps=sensor_fps,
    )
    cam_right = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C,
        sensorFps=sensor_fps,
    )

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setLeftRightCheck(True)

    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=oak_internal_sync_ms))
    sync.setSyncAttempts(-1)

    rgb_out = cam_rgb.requestOutput(
        size=rgb_size,
        type=dai.ImgFrame.Type.BGR888p,
        fps=output_fps,
        enableUndistortion=enable_undistortion,
    )
    left_out = cam_left.requestOutput(size=mono_size, fps=output_fps)
    right_out = cam_right.requestOutput(size=mono_size, fps=output_fps)

    rgb_out.link(sync.inputs["rgb"])
    left_out.link(stereo.left)
    right_out.link(stereo.right)

    if platform == dai.Platform.RVC4:
        align = pipeline.create(dai.node.ImageAlign)
        stereo.depth.link(align.input)
        rgb_out.link(align.inputAlignTo)
        align.outputAligned.link(sync.inputs["depth_aligned"])
    else:
        stereo.depth.link(sync.inputs["depth_aligned"])
        rgb_out.link(stereo.inputAlignTo)

    try:
        q = sync.out.createOutputQueue(maxSize=OAK_QUEUE_MAXSIZE, blocking=False)
    except TypeError:
        q = sync.out.createOutputQueue()
    return q


def open_oaks_v3(
    exit_stack: contextlib.ExitStack,
    oak_ids: List[str],
    output_fps: int,
    sensor_fps: int,
    rgb_size: Tuple[int, int],
    mono_size: Tuple[int, int],
    oak_internal_sync_ms: float,
    enable_undistortion: bool,
) -> Dict[str, Dict]:
    available = set(list_available_oaks())
    missing = [mxid for mxid in oak_ids if mxid not in available]
    if missing:
        raise RuntimeError(f"Requested OAK(s) not detected: {missing}")

    opened: Dict[str, Dict] = {}
    for mxid in oak_ids:
        print(f"Opening specific OAK by MXID: {mxid}")
        device = dai.Device(dai.DeviceInfo(mxid))
        pipeline = exit_stack.enter_context(dai.Pipeline(device))
        opened_mxid = pipeline.getDefaultDevice().getDeviceInfo().getDeviceId()
        if opened_mxid != mxid:
            raise RuntimeError(f"Requested {mxid}, but pipeline opened {opened_mxid}")

        q = build_oak_pipeline_v3(
            pipeline=pipeline,
            output_fps=output_fps,
            sensor_fps=sensor_fps,
            rgb_size=rgb_size,
            mono_size=mono_size,
            oak_internal_sync_ms=oak_internal_sync_ms,
            enable_undistortion=enable_undistortion,
        )
        pipeline.start()
        source_name = f"oak_{mxid}"
        opened[mxid] = {"device": device, "pipeline": pipeline, "queue": q, "source_name": source_name}
        print(f"Opened {source_name}")
    return opened


# -------------------------
# Reader threads
# -------------------------
class SharedMemoryStore:
    def __init__(self, source_names: List[str]):
        self.lock = threading.Lock()
        self.frames: Dict[str, List[MemFrame]] = {name: [] for name in source_names}
        self.latest: Dict[str, Optional[MemFrame]] = {name: None for name in source_names}
        self.raw_counts: Dict[str, int] = {name: 0 for name in source_names}
        self.record_counts: Dict[str, int] = {name: 0 for name in source_names}
        self.errors: Dict[str, List[str]] = {name: [] for name in source_names}

    def append_frame(self, source_name: str, frame: MemFrame, record: bool) -> None:
        with self.lock:
            self.raw_counts[source_name] += 1
            self.latest[source_name] = frame
            if record:
                self.frames[source_name].append(frame)
                self.record_counts[source_name] += 1

    def add_error(self, source_name: str, msg: str) -> None:
        with self.lock:
            self.errors[source_name].append(msg)

    def snapshot_latest(self) -> Dict[str, Optional[MemFrame]]:
        with self.lock:
            return dict(self.latest)

    def get_frames_copy(self) -> Dict[str, List[MemFrame]]:
        with self.lock:
            return {k: list(v) for k, v in self.frames.items()}


def reader_loop(
    source_name: str,
    q,
    projector: DeviceTimeProjector,
    store: SharedMemoryStore,
    stop_evt: threading.Event,
    record_evt: threading.Event,
) -> None:
    while not stop_evt.is_set():
        try:
            msg = q.tryGet()
        except Exception as e:
            store.add_error(source_name, f"queue tryGet failed: {repr(e)}")
            time.sleep(0.005)
            continue

        if msg is None:
            time.sleep(0.001)
            continue

        host_ts = time.monotonic()
        try:
            rgb_msg = msg["rgb"]
            depth_msg = msg["depth_aligned"]
            rgb = rgb_msg.getCvFrame().copy()
            depth = depth_msg.getFrame().copy()
            dev_ts = td_to_sec(rgb_msg.getTimestamp())
            try:
                frame_num = int(rgb_msg.getSequenceNum())
            except Exception:
                frame_num = None
            match_ts = projector.project(host_ts, dev_ts)
            frame = MemFrame(
                source=source_name,
                host_ts=host_ts,
                device_ts=dev_ts,
                match_ts=match_ts,
                frame_number=frame_num,
                rgb=rgb,
                depth=depth,
            )
            store.append_frame(source_name, frame, record=record_evt.is_set())
        except Exception as e:
            store.add_error(source_name, f"message decode failed: {repr(e)}")
            time.sleep(0.002)


# -------------------------
# Offline synchronization
# -------------------------
def _frame_arrays(frames_by_source: Dict[str, List[MemFrame]]) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for name, frames in frames_by_source.items():
        ts = np.asarray([f.match_ts for f in frames], dtype=np.float64)
        out[name] = {"ts": ts}
    return out


def nearest_index_after(ts: np.ndarray, target: float, min_idx: int) -> Optional[int]:
    if ts.size == 0 or min_idx >= ts.size:
        return None
    j = int(np.searchsorted(ts, target, side="left"))
    candidates = []
    for k in (j - 1, j, j + 1):
        if k >= min_idx and 0 <= k < ts.size:
            candidates.append(k)
    if not candidates:
        return None
    return min(candidates, key=lambda k: abs(float(ts[k]) - target))


def evaluate_phase(
    frames_by_source: Dict[str, List[MemFrame]],
    source_names: List[str],
    period_s: float,
    tolerance_s: float,
    start_ts: float,
    end_ts: float,
    phase_s: float,
) -> Tuple[int, float, float]:
    arrays = _frame_arrays(frames_by_source)
    t0 = start_ts + phase_s
    if t0 > end_ts:
        return 0, float("inf"), float("inf")

    n_targets = int(math.floor((end_ts - t0) / period_s)) + 1
    valid = 0
    spans = []
    errors = []
    for i in range(n_targets):
        target = t0 + i * period_s
        picked_ts = []
        ok = True
        for name in source_names:
            ts = arrays[name]["ts"]
            if ts.size == 0:
                ok = False
                break
            j = int(np.searchsorted(ts, target, side="left"))
            cand = []
            for k in (j - 1, j, j + 1):
                if 0 <= k < ts.size:
                    cand.append(k)
            if not cand:
                ok = False
                break
            best = min(cand, key=lambda k: abs(float(ts[k]) - target))
            err = abs(float(ts[best]) - target)
            if err > tolerance_s:
                ok = False
                break
            picked_ts.append(float(ts[best]))
            errors.append(err)
        if ok:
            valid += 1
            spans.append(max(picked_ts) - min(picked_ts))
    if valid == 0:
        return 0, float("inf"), float("inf")
    return valid, float(np.median(spans)), float(np.median(errors))


def choose_best_phase(
    frames_by_source: Dict[str, List[MemFrame]],
    source_names: List[str],
    sync_fps: float,
    tolerance_s: float,
    phase_steps: int,
) -> Tuple[float, Dict]:
    period_s = 1.0 / sync_fps
    firsts = []
    lasts = []
    for name in source_names:
        if not frames_by_source[name]:
            raise RuntimeError(f"No recorded frames for {name}")
        firsts.append(frames_by_source[name][0].match_ts)
        lasts.append(frames_by_source[name][-1].match_ts)
    start_ts = max(firsts)
    end_ts = min(lasts)
    if end_ts <= start_ts:
        raise RuntimeError("No overlapping time range among cameras.")

    # Try different grid phases within one output period. Prefer more valid timesteps,
    # then smaller median within-bundle span, then smaller median target error.
    best = None
    phases = np.linspace(0.0, period_s, phase_steps, endpoint=False)
    for phase in phases:
        valid, med_span, med_err = evaluate_phase(
            frames_by_source, source_names, period_s, tolerance_s, start_ts, end_ts, float(phase)
        )
        score = (valid, -med_span, -med_err)
        if best is None or score > best[0]:
            best = (score, float(phase), valid, med_span, med_err)

    assert best is not None
    _, phase_s, valid, med_span, med_err = best
    meta = {
        "overlap_start_ts": start_ts,
        "overlap_end_ts": end_ts,
        "period_s": period_s,
        "chosen_phase_s": phase_s,
        "phase_steps": phase_steps,
        "estimated_valid_timesteps_for_phase": valid,
        "estimated_median_span_ms_for_phase": med_span * 1000.0 if np.isfinite(med_span) else None,
        "estimated_median_target_error_ms_for_phase": med_err * 1000.0 if np.isfinite(med_err) else None,
    }
    return start_ts + phase_s, meta


def offline_fixed_rate_sync(
    frames_by_source: Dict[str, List[MemFrame]],
    source_names: List[str],
    sync_fps: float,
    tolerance_ms: float,
    phase_steps: int,
    allow_reuse: bool = False,
) -> Tuple[List[Dict[str, MemFrame]], Dict]:
    tolerance_s = tolerance_ms / 1000.0
    period_s = 1.0 / sync_fps
    arrays = _frame_arrays(frames_by_source)

    start_target_ts, phase_meta = choose_best_phase(
        frames_by_source, source_names, sync_fps, tolerance_s, phase_steps=phase_steps
    )
    end_ts = phase_meta["overlap_end_ts"]
    n_targets = int(math.floor((end_ts - start_target_ts) / period_s)) + 1

    last_idx = {name: 0 for name in source_names}
    matched_groups: List[Dict[str, MemFrame]] = []
    dropped_targets = []
    target_errors_ms = []
    spans_ms = []

    for ti in range(n_targets):
        target = start_target_ts + ti * period_s
        group = {}
        indices = {}
        ok = True
        fail_reason = ""

        for name in source_names:
            ts = arrays[name]["ts"]
            min_idx = 0 if allow_reuse else last_idx[name]
            idx = nearest_index_after(ts, target, min_idx=min_idx)
            if idx is None:
                ok = False
                fail_reason = f"no candidate for {name}"
                break
            err_s = abs(float(ts[idx]) - target)
            if err_s > tolerance_s:
                ok = False
                fail_reason = f"{name} nearest error {err_s*1000.0:.2f}ms > tolerance"
                break
            indices[name] = idx
            group[name] = frames_by_source[name][idx]

        if ok:
            if not allow_reuse:
                for name, idx in indices.items():
                    last_idx[name] = idx + 1
            matched_groups.append(group)
            picked_ts = [fr.match_ts for fr in group.values()]
            spans_ms.append(1000.0 * (max(picked_ts) - min(picked_ts)))
            target_errors_ms.extend([1000.0 * (fr.match_ts - target) for fr in group.values()])
        else:
            dropped_targets.append({"target_index": ti, "target_ts": target, "reason": fail_reason})
            # Important: do not advance last_idx on failure. A late frame can still be used
            # by a future target if it is closest and within tolerance.

    meta = {
        **phase_meta,
        "sync_fps": sync_fps,
        "sync_tolerance_ms": tolerance_ms,
        "num_target_timesteps": n_targets,
        "num_saved_timesteps": len(matched_groups),
        "num_dropped_timesteps": len(dropped_targets),
        "allow_reuse": allow_reuse,
        "dropped_targets_preview": dropped_targets[:100],
    }
    if spans_ms:
        arr = np.asarray(spans_ms)
        meta.update(
            {
                "match_span_ms_min": float(arr.min()),
                "match_span_ms_p50": float(np.percentile(arr, 50)),
                "match_span_ms_p90": float(np.percentile(arr, 90)),
                "match_span_ms_p95": float(np.percentile(arr, 95)),
                "match_span_ms_max": float(arr.max()),
            }
        )
    if target_errors_ms:
        arr = np.asarray(target_errors_ms)
        meta.update(
            {
                "target_error_ms_abs_p50": float(np.percentile(np.abs(arr), 50)),
                "target_error_ms_abs_p90": float(np.percentile(np.abs(arr), 90)),
                "target_error_ms_abs_p95": float(np.percentile(np.abs(arr), 95)),
                "target_error_ms_abs_max": float(np.max(np.abs(arr))),
            }
        )
    return matched_groups, meta


# -------------------------
# Saving
# -------------------------
def save_bundle(bundle_idx: int, out_root: Path, matched: Dict[str, MemFrame], extra_meta: Dict) -> None:
    bundle_dir = out_root / f"bundle_{bundle_idx:06d}"
    ensure_dir(bundle_dir)

    match_ts_values = [fr.match_ts for fr in matched.values()]
    host_ts_values = [fr.host_ts for fr in matched.values()]
    min_match_ts = min(match_ts_values)
    min_host_ts = min(host_ts_values)

    meta = {
        "bundle_index": bundle_idx,
        "saved_at_wallclock": datetime.now().isoformat(),
        "match_span_ms": 1000.0 * (max(match_ts_values) - min_match_ts),
        "host_span_ms": 1000.0 * (max(host_ts_values) - min_host_ts),
        **extra_meta,
        "sources": {},
    }

    for source_name, frame in sorted(matched.items()):
        source_dir = bundle_dir / source_name
        ensure_dir(source_dir)
        cv2.imwrite(
            str(source_dir / "rgb.jpg"),
            frame.rgb,
            [int(cv2.IMWRITE_JPEG_QUALITY), RGB_JPEG_QUALITY],
        )
        np.save(source_dir / "depth.npy", frame.depth)
        meta["sources"][source_name] = {
            "host_ts": frame.host_ts,
            "device_ts": frame.device_ts,
            "match_ts": frame.match_ts,
            "match_ts_offset_from_group_min_ms": 1000.0 * (frame.match_ts - min_match_ts),
            "host_ts_offset_from_group_min_ms": 1000.0 * (frame.host_ts - min_host_ts),
            "frame_number": frame.frame_number,
            "rgb_shape": list(frame.rgb.shape),
            "depth_shape": list(frame.depth.shape),
            "depth_dtype": str(frame.depth.dtype),
        }

    with open(bundle_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_all_bundles(
    out_root: Path,
    groups: List[Dict[str, MemFrame]],
    args_meta: Dict,
    sync_meta: Dict,
    projectors_meta: Dict[str, Dict],
) -> None:
    ensure_dir(out_root)
    print(f"Saving {len(groups)} synchronized bundles to {out_root} ...")
    t0 = time.monotonic()
    for i, group in enumerate(groups):
        picked_ts = [fr.match_ts for fr in group.values()]
        target_ts = None
        if sync_meta.get("period_s") is not None:
            target_ts = sync_meta["overlap_start_ts"] + sync_meta["chosen_phase_s"] + i * sync_meta["period_s"]
        extra_meta = {
            **args_meta,
            "offline_sync": True,
            "target_ts_estimate": target_ts,
            "per_source_match_offsets_ms": {
                name: 1000.0 * (fr.match_ts - min(picked_ts)) for name, fr in group.items()
            },
            "projectors": projectors_meta,
        }
        save_bundle(i, out_root, group, extra_meta)
        if i % 20 == 0:
            print(f"  saved bundle {i:06d}/{len(groups)}")
    dt = time.monotonic() - t0
    print(f"Finished saving bundles in {dt:.2f}s")


# -------------------------
# Preview
# -------------------------
def update_preview(store: SharedMemoryStore, scale: float) -> bool:
    latest = store.snapshot_latest()
    if not latest:
        return True
    panels = []
    for source_name, frame in sorted(latest.items()):
        if frame is None:
            continue
        rgb = maybe_resize(frame.rgb, scale)
        cv2.putText(rgb, source_name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        panels.append(rgb)
    if panels:
        try:
            view = np.hstack(panels)
            cv2.imshow("OAK RGB preview", view)
            key = cv2.waitKey(1)
            if key in (ord("q"), 27):
                return False
        except Exception:
            pass
    return True


# -------------------------
# CLI and main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DepthAI v3 3-OAK memory-first recorder with offline synchronization.")
    p.add_argument("--oak-ids", nargs="+", default=DEFAULT_OAK_IDS, help="OAK MXIDs; defaults to $AURA_OAK_IDS, camera-index order.")
    p.add_argument("--resolution", choices=sorted(RESOLUTION_PRESETS.keys()), default="high")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help="DepthAI output FPS.")
    p.add_argument("--sensor-fps", type=int, default=DEFAULT_SENSOR_FPS, help="OAK sensor FPS. Keep at 15 if 10 FPS is black.")
    p.add_argument("--sync-fps", type=float, default=DEFAULT_SYNC_FPS, help="Offline fixed-rate synchronized output FPS.")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="Recording duration in seconds. Hard capped at 30.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--warmup-seconds", type=float, default=4.0)
    p.add_argument("--preview", type=str2bool, default=False)
    p.add_argument("--preview-scale", type=float, default=PREVIEW_SCALE)
    p.add_argument("--sync-tolerance-ms", type=float, default=DEFAULT_SYNC_TOLERANCE_MS)
    p.add_argument("--oak-internal-sync-ms", type=float, default=DEFAULT_INTERNAL_SYNC_MS)
    p.add_argument("--enable-undistortion", type=str2bool, default=True)
    p.add_argument("--freeze-percentile", type=float, default=DEFAULT_OFFSET_FREEZE_PERCENTILE)
    p.add_argument("--phase-steps", type=int, default=120, help="How many grid phases to test within one output period.")
    p.add_argument("--allow-reuse", type=str2bool, default=False, help="Allow reusing a camera frame for adjacent output timesteps. Not recommended for dynamics.")
    p.add_argument("--overwrite", action="store_true", help="Allow an output folder that already contains bundle_* or recording_summary.json. Nothing is deleted: stale bundle_* folders from an earlier, longer run remain.")
    args = p.parse_args()
    if not args.oak_ids:
        p.error("no OAK camera ids: set AURA_OAK_IDS in config/aura.env and load it (set -a; source config/aura.env; set +a) or pass --oak-ids <MXID> [<MXID> ...]")
    return args


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.duration > MAX_DURATION_S:
        raise ValueError(f"For safety, --duration is capped at {MAX_DURATION_S:.1f}s. You requested {args.duration:.1f}s.")

    preset = RESOLUTION_PRESETS[args.resolution]
    rgb_size = preset["rgb"]
    mono_size = preset["mono"]
    source_names = [f"oak_{mxid}" for mxid in args.oak_ids]

    if args.output_root.exists() and not args.overwrite:
        existing = list(args.output_root.glob("bundle_*"))
        if existing or (args.output_root / "recording_summary.json").exists():
            raise RuntimeError(
                f"Output folder {args.output_root} already contains recording files. "
                f"Use --overwrite or choose a new folder."
            )
    ensure_dir(args.output_root)

    avail_gib = get_mem_available_gib()
    est_gib = estimate_ram_gib(len(args.oak_ids), rgb_size, args.fps, args.duration)

    print("========== DepthAI v3 OAK RGB-D recorder: memory-first offline sync ==========")
    print(f"DepthAI version: {dai.__version__}")
    print(f"Recording OAK IDs: {args.oak_ids}")
    print(f"Output: {args.output_root.resolve()}")
    print(f"Resolution: {args.resolution} RGB {rgb_size} mono/depth {mono_size}")
    print(f"Output FPS: {args.fps}")
    print(f"Sensor FPS: {args.sensor_fps}")
    print(f"Offline sync FPS: {args.sync_fps}")
    print(f"Duration: {args.duration:.2f}s / max {MAX_DURATION_S:.1f}s")
    print(f"Warmup: {args.warmup_seconds:.2f}s")
    print(f"Preview: {args.preview}")
    print(f"Sync tolerance ms: {args.sync_tolerance_ms}")
    print(f"OAK internal RGB-depth sync ms: {args.oak_internal_sync_ms}")
    print(f"RGB undistortion: {args.enable_undistortion}")
    print(f"Estimated RAM for raw frames: ~{est_gib:.1f} GiB")
    if avail_gib is not None:
        print(f"Current MemAvailable: {avail_gib:.1f} GiB")
        if est_gib > 0.75 * avail_gib:
            print("WARNING: estimated memory is high compared with available RAM.")
    print("===========================================================================")

    stop_evt = threading.Event()
    record_evt = threading.Event()

    with contextlib.ExitStack() as exit_stack:
        oaks = open_oaks_v3(
            exit_stack=exit_stack,
            oak_ids=args.oak_ids,
            output_fps=args.fps,
            sensor_fps=args.sensor_fps,
            rgb_size=rgb_size,
            mono_size=mono_size,
            oak_internal_sync_ms=args.oak_internal_sync_ms,
            enable_undistortion=args.enable_undistortion,
        )

        projectors = {info["source_name"]: DeviceTimeProjector(info["source_name"]) for info in oaks.values()}
        store = SharedMemoryStore(source_names)

        threads = []
        for info in oaks.values():
            name = info["source_name"]
            t = threading.Thread(
                target=reader_loop,
                args=(name, info["queue"], projectors[name], store, stop_evt, record_evt),
                daemon=True,
            )
            t.start()
            threads.append(t)

        print(f"Warming up for {args.warmup_seconds:.2f}s...")
        warmup_end = time.monotonic() + args.warmup_seconds
        try:
            while time.monotonic() < warmup_end:
                if args.preview:
                    if not update_preview(store, args.preview_scale):
                        print("Preview quit during warmup.")
                        stop_evt.set()
                        break
                time.sleep(0.01)

            print("Freezing device->host timestamp offsets:")
            for name, prj in projectors.items():
                prj.freeze(args.freeze_percentile)
                print(f"  {name}: fixed_offset={prj.fixed_offset:.6f}, samples={len(prj.offset_samples)}")

            print("Start MEMORY recording. No online sync/save decisions will be made.")
            start_wall = datetime.now().isoformat()
            record_evt.set()
            record_start = time.monotonic()
            record_end = record_start + args.duration

            last_report = record_start
            while time.monotonic() < record_end and not stop_evt.is_set():
                if args.preview:
                    if not update_preview(store, args.preview_scale):
                        print("Preview quit. Stopping early.")
                        break
                now = time.monotonic()
                if now - last_report >= 1.0:
                    with store.lock:
                        counts = dict(store.record_counts)
                    elapsed = now - record_start
                    mem = get_mem_available_gib()
                    mem_s = f", MemAvailable={mem:.1f}GiB" if mem is not None else ""
                    print(f"  t={elapsed:5.1f}s recorded_counts={counts}{mem_s}")
                    last_report = now
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received. Stopping recording...")
        finally:
            record_evt.clear()
            stop_evt.set()
            for t in threads:
                t.join(timeout=2.0)
            if args.preview:
                cv2.destroyAllWindows()

    frames_by_source = store.get_frames_copy()
    print("Recording stopped. Raw frame counts:")
    for name in source_names:
        print(f"  {name}: {len(frames_by_source[name])} frames")

    # Basic sanity sort by match_ts in case threads appended out of order.
    for name in source_names:
        frames_by_source[name].sort(key=lambda f: f.match_ts)

    print("Running offline fixed-rate synchronization...")
    groups, sync_meta = offline_fixed_rate_sync(
        frames_by_source=frames_by_source,
        source_names=source_names,
        sync_fps=args.sync_fps,
        tolerance_ms=args.sync_tolerance_ms,
        phase_steps=args.phase_steps,
        allow_reuse=args.allow_reuse,
    )

    print("Offline sync result:")
    print(f"  target timesteps : {sync_meta['num_target_timesteps']}")
    print(f"  saved timesteps  : {sync_meta['num_saved_timesteps']}")
    print(f"  dropped timesteps: {sync_meta['num_dropped_timesteps']}")
    print(f"  chosen phase s   : {sync_meta['chosen_phase_s']:.6f}")
    print(f"  match span p50/p95/max ms: {sync_meta.get('match_span_ms_p50')} / {sync_meta.get('match_span_ms_p95')} / {sync_meta.get('match_span_ms_max')}")

    args_meta = {
        "requested_oak_ids": args.oak_ids,
        "requested_oak_rgb_resolution": list(rgb_size),
        "requested_oak_mono_resolution": list(mono_size),
        "requested_output_fps": args.fps,
        "requested_sensor_fps": args.sensor_fps,
        "requested_sync_fps": args.sync_fps,
        "requested_duration_s": args.duration,
        "enable_undistortion": args.enable_undistortion,
        "sync_tolerance_ms": args.sync_tolerance_ms,
        "memory_first_recording": True,
        "offline_sync_mode": "fixed-rate",
    }
    projectors_meta = {name: prj.as_meta() for name, prj in projectors.items()}

    save_all_bundles(
        out_root=args.output_root,
        groups=groups,
        args_meta=args_meta,
        sync_meta=sync_meta,
        projectors_meta=projectors_meta,
    )

    # Summary file.
    summary = {
        "created_at_wallclock": datetime.now().isoformat(),
        "record_start_wallclock": start_wall,
        "depthai_version": dai.__version__,
        "args": vars(args) | {"output_root": str(args.output_root)},
        "raw_frame_counts": {name: len(frames_by_source[name]) for name in source_names},
        "record_counts": store.record_counts,
        "raw_counts_including_warmup": store.raw_counts,
        "reader_errors": store.errors,
        "projectors": projectors_meta,
        "offline_sync": sync_meta,
    }
    with open(args.output_root / "recording_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Saved {len(groups)} synchronized bundles to {args.output_root.resolve()}")
    print(f"Summary: {args.output_root / 'recording_summary.json'}")


if __name__ == "__main__":
    main()
