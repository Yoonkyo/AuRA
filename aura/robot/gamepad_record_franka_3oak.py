#!/usr/bin/env python3
r"""
Multiprocess PS-controller teleoperation + memory-first 3-OAK RGB-D + FR3 state recording.

This script is the OAK-only, gamepad-controlled version of the memory-first
recording idea from aura/capture/record_3oak.py:

  1. Recorder process owns all OAK cameras.
  2. Gamepad process only reads pygame at ~100 Hz and writes a shared 6-DoF command.
  3. Robot process owns the FR3 arm connection and runs the direct 1 kHz active control loop.
  4. Gripper process owns the separate gripper connection; D-pad up toggles open/grasp.
  5. Press Square to start recording.
  6. While recording, every OAK RGB-D frame is stored in CPU RAM.
  7. Robot states are sampled from the already-received 1 kHz FCI state (no get_joints polling).
  8. Press Square again, or hit --max-duration, to stop recording.
  9. After recording stops, OAK frames are synchronized offline at fixed FPS.
 10. Robot joints are linearly interpolated to the actual selected camera capture timestamp.
 11. Synchronized bundles and raw robot states are saved to disk.

Important:
- This is software timestamp synchronization, not hardware trigger sync.
- High resolution uses much more RAM. The script estimates RAM before recording.
- Only ONE FR3 arm connection is open at a time: the dedicated robot process owns it
  (the gripper process owns the separate gripper connection).
- --home-on-start defaults to True: the robot moves to its home pose right after
  launch, without a prompt. Pass --home-on-start False to skip homing.
- Stop with Square or Circle, not Ctrl-C: on Ctrl-C the recorder exits without
  saving the frames held in RAM.
- The gamepad input filter, the 1 kHz jerk-limited velocity smoother and the pose integration
  are adapted from the fr3 library (third_party/fr3: Gamepad.update in _gamepad.py,
  ControlThread._velocity_loop in _control.py); the stick mapping, deadzone, speed limits and
  gripper settings follow fr3's examples/07_gamepad_control.py, which is not vendored. These
  parts fall under fr3's license, which is still to be confirmed (see THIRD_PARTY_NOTICES.md).
- Many camera, sync and saving helpers started as copies of aura/capture/record_3oak.py
  (they are not imported), and several have since diverged on purpose: e.g. reader_loop
  here matches frames on the DepthAI capture timestamp, not on the DeviceTimeProjector
  output (see the comment there). Do not re-sync the two files blindly.

Controls
--------
Left stick Y/X      x / y velocity
Right stick X/Y     roll / pitch rotation
LT                  z down  (LB + LT = z up)
RT                  yaw -   (RB + RT = yaw +)
Cross               disabled (prints a notice; homing only via --home-on-start)
Circle              stop/quit; if recording, recorder saves the current data
D-pad up            toggle gripper
Square              start / stop recording

Example
-------
From the repository root, with config/aura.env loaded (set -a; source config/aura.env; set +a);
the robot IP defaults to $AURA_ROBOT_IP and the cameras to $AURA_OAK_IDS (camera-index order):

uv run --project "$AURA_ROBOT_PROJECT" python aura/robot/gamepad_record_franka_3oak.py \
  --resolution high --max-duration 30 --output-root "$AURA_DATA_ROOT/<session>"

To override them, pass <ROBOT_IP> as the first argument and/or --oak-ids <MXID_0> <MXID_1> <MXID_2>.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import math
import multiprocessing as mp
import os
import queue
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
    raise RuntimeError("Failed to import depthai. Install/activate the environment with DepthAI v3.") from e

import fr3


# -------------------------
# Defaults
# -------------------------
DEFAULT_ROBOT_IP = os.environ.get("AURA_ROBOT_IP")  # config/aura.env; checked in validate_args()
DEFAULT_OAK_IDS = os.environ.get("AURA_OAK_IDS", "").split()  # MXIDs, camera-index order

RESOLUTION_PRESETS = {
    "low": {"rgb": (640, 400), "mono": (640, 400)},
    "high": {"rgb": (1280, 720), "mono": (1280, 720)},
}

DEFAULT_FPS = 15
DEFAULT_SENSOR_FPS = 15
DEFAULT_SYNC_FPS = 15.0
DEFAULT_SYNC_TOLERANCE_MS = 45.0
DEFAULT_INTERNAL_SYNC_MS = 25.0
DEFAULT_WARMUP_SECONDS = 4.0
DEFAULT_MAX_DURATION_S = 30.0
HARD_MAX_DURATION_S = 60.0
DEFAULT_OFFSET_FREEZE_PERCENTILE = 5.0
DEFAULT_PHASE_STEPS = 120
OAK_QUEUE_MAXSIZE = 16
RGB_JPEG_QUALITY = 90
PREVIEW_SCALE = 0.35

MAX_LIN = 0.12
MAX_ANG = 0.35
DZ = 0.10

# PS (DualShock-style) controller mapping as enumerated by pygame on Linux; other pads may differ.
PS_CROSS = 0
PS_CIRCLE = 1
PS_SQUARE = 3
PS_LB = 4
PS_RB = 5

GRIPPER_SPEED = 0.1
GRIPPER_GRASP_WIDTH = 0.0
GRIPPER_GRASP_FORCE = 20.0
GRIPPER_GRASP_EPSILON = 0.08

STATE_PUBLISH_HZ = 100.0  # Sample already-received FCI states; no extra robot API call.
MAIN_STATUS_POLL_TIMEOUT_S = 0.25

SAVE_PREFIX = "recording_sync_robot"


# -------------------------
# Helpers
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
    raw_bytes_per_frame_per_cam = w * h * (3 + 2)  # RGB uint8 + depth uint16
    raw = raw_bytes_per_frame_per_cam * num_cams * fps * duration_s
    return raw * 1.6 / (1024.0 ** 3)  # conservative Python overhead estimate


def status_put(status_q: mp.Queue, msg: str) -> None:
    try:
        status_q.put_nowait(msg)
    except Exception:
        pass


# -------------------------
# Frame / projector types
# -------------------------
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
    """Project OAK device timestamps into a common host monotonic-time domain."""

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
def list_available_oaks(status_q: Optional[mp.Queue] = None) -> List[str]:
    devices = dai.Device.getAllAvailableDevices()
    ids = []
    msg_lines = ["Available OAK devices:"]
    for d in devices:
        mxid = d.getDeviceId()
        ids.append(mxid)
        msg_lines.append(f"  {mxid}  state={getattr(d, 'state', 'unknown')}")
    for line in msg_lines:
        if status_q is not None:
            status_put(status_q, line)
        else:
            print(line)
    return ids


def build_oak_pipeline_v3(
    pipeline: dai.Pipeline,
    output_fps: int,
    sensor_fps: int,
    rgb_size: Tuple[int, int],
    mono_size: Tuple[int, int],
    oak_internal_sync_ms: float,
    enable_undistortion: bool,
    status_q: Optional[mp.Queue] = None,
):
    device = pipeline.getDefaultDevice()
    platform = device.getPlatform()
    mxid = device.getDeviceInfo().getDeviceId()
    if status_q is not None:
        status_put(status_q, f"[{mxid}] platform={platform}")
        try:
            status_put(status_q, f"[{mxid}] USB speed={device.getUsbSpeed()}")
        except Exception:
            pass
        status_put(
            status_q,
            f"[{mxid}] RGB size={rgb_size}, mono/depth size={mono_size}, "
            f"sensor_fps={sensor_fps}, output_fps={output_fps}, undistort={enable_undistortion}",
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
    status_q: mp.Queue,
) -> Dict[str, Dict]:
    available = set(list_available_oaks(status_q))
    missing = [mxid for mxid in oak_ids if mxid not in available]
    if missing:
        raise RuntimeError(f"Requested OAK(s) not detected: {missing}")

    opened: Dict[str, Dict] = {}
    for mxid in oak_ids:
        status_put(status_q, f"Opening specific OAK by MXID: {mxid}")
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
            status_q=status_q,
        )
        pipeline.start()
        source_name = f"oak_{mxid}"
        opened[mxid] = {"device": device, "pipeline": pipeline, "queue": q, "source_name": source_name}
        status_put(status_q, f"Opened {source_name}")
    return opened


# -------------------------
# In-memory camera store
# -------------------------
class SharedMemoryStore:
    def __init__(self, source_names: List[str]):
        self.lock = threading.Lock()
        self.frames: Dict[str, List[MemFrame]] = {name: [] for name in source_names}
        self.latest: Dict[str, Optional[MemFrame]] = {name: None for name in source_names}
        self.raw_counts: Dict[str, int] = {name: 0 for name in source_names}
        self.record_counts: Dict[str, int] = {name: 0 for name in source_names}
        self.errors: Dict[str, List[str]] = {name: [] for name in source_names}

    def clear_recorded(self) -> None:
        with self.lock:
            for name in self.frames:
                self.frames[name].clear()
                self.record_counts[name] = 0

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

    def get_record_counts(self) -> Dict[str, int]:
        with self.lock:
            return dict(self.record_counts)


def reader_loop(
    source_name: str,
    q,
    projector: DeviceTimeProjector,
    store: SharedMemoryStore,
    stop_evt: threading.Event,
    record_flag,
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
            # DepthAI getTimestamp() is already synchronized to dai.Clock.now(),
            # i.e. the host monotonic clock domain. Do NOT add a receive-latency
            # offset to it. The old code fed this timestamp through
            # DeviceTimeProjector, which shifted camera capture times by USB/
            # pipeline latency and caused robot/camera temporal misalignment.
            dev_ts = td_to_sec(rgb_msg.getTimestamp())
            try:
                frame_num = int(rgb_msg.getSequenceNum())
            except Exception:
                frame_num = None

            # Keep projector statistics as diagnostics only (host receive time -
            # capture timestamp), but use the host-synchronized capture timestamp
            # itself for all matching/synchronization decisions.
            _ = projector.project(host_ts, dev_ts)
            match_ts = dev_ts
            frame = MemFrame(
                source=source_name,
                host_ts=host_ts,
                device_ts=dev_ts,
                match_ts=match_ts,
                frame_number=frame_num,
                rgb=rgb,
                depth=depth,
            )
            store.append_frame(source_name, frame, record=bool(record_flag.value))
        except Exception as e:
            store.add_error(source_name, f"message decode failed: {repr(e)}")
            time.sleep(0.002)


# -------------------------
# Robot states: shared fixed-capacity buffer
# -------------------------
def robot_state_capacity(max_duration_s: float, state_hz: float) -> int:
    # A little headroom for start/end samples.
    return max(64, int(math.ceil(max_duration_s * state_hz)) + 32)


def reset_robot_state_buffer(state_count) -> None:
    state_count.value = 0


def append_robot_state_shared(
    state_ts,
    state_q,
    state_success,
    state_count,
    capacity: int,
    host_ts: float,
    joints,
    success_rate: float,
) -> bool:
    """Append one already-received FCI state without any robot API call."""
    idx = int(state_count.value)
    if idx >= capacity:
        return False
    state_ts[idx] = float(host_ts)
    base = idx * 7
    for j in range(7):
        state_q[base + j] = float(joints[j])
    state_success[idx] = float(success_rate)
    # Publish count last so readers never see a partially written row.
    state_count.value = idx + 1
    return True


def snapshot_robot_states(state_ts, state_q, state_success, state_count) -> List[Dict]:
    n = int(state_count.value)
    states: List[Dict] = []
    for i in range(n):
        base = i * 7
        states.append(
            {
                "state_seq": int(i),
                "state_type": "fci_joint_angles_sample",
                "host_ts": float(state_ts[i]),
                "read_start_ts": float(state_ts[i]),
                "read_end_ts": float(state_ts[i]),
                "read_duration_ms": 0.0,
                "joint_angles_rad": [float(state_q[base + j]) for j in range(7)],
                "control_command_success_rate": float(state_success[i]),
            }
        )
    return states


def save_raw_robot_states_npz(out_root: Path, robot_states: List[Dict]) -> Optional[Path]:
    if not robot_states:
        return None
    ts = np.asarray([s["host_ts"] for s in robot_states], dtype=np.float64)
    q = np.asarray([s["joint_angles_rad"] for s in robot_states], dtype=np.float64)
    success = np.asarray(
        [s.get("control_command_success_rate", np.nan) for s in robot_states],
        dtype=np.float64,
    )
    path = out_root / "robot_states_raw.npz"
    np.savez_compressed(
        path,
        host_ts=ts,
        joint_angles_rad=q,
        control_command_success_rate=success,
    )
    return path


def robot_state_stats(robot_states: List[Dict]) -> Dict:
    meta = {"num_robot_states": len(robot_states)}
    if not robot_states:
        return meta
    ts = np.asarray([float(s["host_ts"]) for s in robot_states], dtype=np.float64)
    success = np.asarray(
        [float(s.get("control_command_success_rate", np.nan)) for s in robot_states],
        dtype=np.float64,
    )
    meta.update(
        {
            "first_robot_state_ts": float(ts[0]),
            "last_robot_state_ts": float(ts[-1]),
            "robot_state_duration_s": float(ts[-1] - ts[0]) if len(ts) >= 2 else 0.0,
        }
    )
    if len(ts) >= 2:
        gaps_ms = np.diff(ts) * 1000.0
        meta.update(
            {
                "robot_state_gap_ms_p50": float(np.percentile(gaps_ms, 50)),
                "robot_state_gap_ms_p95": float(np.percentile(gaps_ms, 95)),
                "robot_state_gap_ms_max": float(np.max(gaps_ms)),
            }
        )
    good = success[np.isfinite(success)]
    if good.size:
        meta.update(
            {
                "control_command_success_rate_min": float(np.min(good)),
                "control_command_success_rate_p50": float(np.percentile(good, 50)),
            }
        )
    return meta


def interpolate_robot_state(robot_states: List[Dict], target_ts: float) -> Optional[Dict]:
    """Linearly interpolate joints to an actual camera capture timestamp."""
    if not robot_states:
        return None

    ts = np.asarray([float(s["host_ts"]) for s in robot_states], dtype=np.float64)
    q = np.asarray([s["joint_angles_rad"] for s in robot_states], dtype=np.float64)
    j = int(np.searchsorted(ts, target_ts, side="left"))

    # Outside the recorded range: attach nearest endpoint and mark extrapolation.
    if j <= 0:
        idx = 0
        state = dict(robot_states[idx])
        signed_ms = 1000.0 * (float(ts[idx]) - float(target_ts))
        state.update(
            {
                "state_type": "joint_angles_nearest_endpoint",
                "sync_target_ts": float(target_ts),
                "sync_error_from_target_ms": float(signed_ms),
                "sync_error_abs_ms": float(abs(signed_ms)),
                "nearest_robot_state_index": int(idx),
                "interpolation_extrapolated": True,
            }
        )
        return state
    if j >= len(ts):
        idx = len(ts) - 1
        state = dict(robot_states[idx])
        signed_ms = 1000.0 * (float(ts[idx]) - float(target_ts))
        state.update(
            {
                "state_type": "joint_angles_nearest_endpoint",
                "sync_target_ts": float(target_ts),
                "sync_error_from_target_ms": float(signed_ms),
                "sync_error_abs_ms": float(abs(signed_ms)),
                "nearest_robot_state_index": int(idx),
                "interpolation_extrapolated": True,
            }
        )
        return state

    i0, i1 = j - 1, j
    t0, t1 = float(ts[i0]), float(ts[i1])
    gap = max(t1 - t0, 1e-12)
    alpha = min(1.0, max(0.0, (float(target_ts) - t0) / gap))
    qi = (1.0 - alpha) * q[i0] + alpha * q[i1]
    nearest_err_ms = 1000.0 * min(abs(float(target_ts) - t0), abs(t1 - float(target_ts)))

    return {
        "state_seq": None,
        "state_type": "joint_angles_interpolated",
        "host_ts": float(target_ts),
        "joint_angles_rad": qi.tolist(),
        "sync_target_ts": float(target_ts),
        "sync_error_from_target_ms": 0.0,
        # Keep this comparable with the old summary: distance to nearest raw state.
        "sync_error_abs_ms": float(nearest_err_ms),
        "interpolation_extrapolated": False,
        "interpolation_alpha": float(alpha),
        "interpolation_left_index": int(i0),
        "interpolation_right_index": int(i1),
        "interpolation_left_ts": t0,
        "interpolation_right_ts": t1,
        "interpolation_bracket_gap_ms": float(1000.0 * (t1 - t0)),
        "control_command_success_rate": float(
            min(
                robot_states[i0].get("control_command_success_rate", 1.0),
                robot_states[i1].get("control_command_success_rate", 1.0),
            )
        ),
    }


# -------------------------
# Offline camera synchronization
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
) -> Tuple[List[Dict[str, MemFrame]], List[float], Dict]:
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
    target_times: List[float] = []
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
            target_times.append(target)
            picked_ts = [fr.match_ts for fr in group.values()]
            spans_ms.append(1000.0 * (max(picked_ts) - min(picked_ts)))
            target_errors_ms.extend([1000.0 * (fr.match_ts - target) for fr in group.values()])
        else:
            dropped_targets.append({"target_index": ti, "target_ts": target, "reason": fail_reason})

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
    return matched_groups, target_times, meta


# -------------------------
# Saving
# -------------------------
def save_bundle(
    bundle_idx: int,
    out_root: Path,
    matched: Dict[str, MemFrame],
    target_ts: float,
    robot_state: Optional[Dict],
    extra_meta: Dict,
) -> None:
    bundle_dir = out_root / f"bundle_{bundle_idx:06d}"
    ensure_dir(bundle_dir)

    match_ts_values = [fr.match_ts for fr in matched.values()]
    host_ts_values = [fr.host_ts for fr in matched.values()]
    min_match_ts = min(match_ts_values)
    min_host_ts = min(host_ts_values)

    meta = {
        "bundle_index": bundle_idx,
        "saved_at_wallclock": datetime.now().isoformat(),
        "offline_sync": True,
        "target_ts": float(target_ts),
        "match_span_ms": 1000.0 * (max(match_ts_values) - min_match_ts),
        "host_span_ms": 1000.0 * (max(host_ts_values) - min_host_ts),
        "robot": robot_state if robot_state is not None else {"error": "No robot state recorded."},
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
            "capture_ts_host_synced": frame.match_ts,
            "match_ts": frame.match_ts,
            "match_ts_offset_from_group_min_ms": 1000.0 * (frame.match_ts - min_match_ts),
            "host_ts_offset_from_group_min_ms": 1000.0 * (frame.host_ts - min_host_ts),
            "target_error_ms": 1000.0 * (frame.match_ts - target_ts),
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
    target_times: List[float],
    robot_states: List[Dict],
    args_meta: Dict,
    sync_meta: Dict,
    projectors_meta: Dict[str, Dict],
    status_q: Optional[mp.Queue] = None,
) -> Dict:
    ensure_dir(out_root)
    msg = f"Saving {len(groups)} synchronized bundles to {out_root} ..."
    status_put(status_q, msg) if status_q is not None else print(msg)
    t0 = time.monotonic()

    robot_errors = []
    for i, (group, target_ts) in enumerate(zip(groups, target_times)):
        # target_ts is only the fixed-rate grid used to choose a camera group.
        # Align robot state to the ACTUAL selected camera capture time instead.
        # For multiple cameras, the median is robust to a small inter-camera span.
        robot_target_ts = float(np.median([fr.match_ts for fr in group.values()]))
        robot = interpolate_robot_state(robot_states, robot_target_ts)
        if robot is not None and "sync_error_abs_ms" in robot:
            robot_errors.append(float(robot["sync_error_abs_ms"]))
        extra_meta = {
            **args_meta,
            "projectors": projectors_meta,
            "offline_sync_meta_preview": {
                "sync_fps": sync_meta.get("sync_fps"),
                "sync_tolerance_ms": sync_meta.get("sync_tolerance_ms"),
                "period_s": sync_meta.get("period_s"),
                "chosen_phase_s": sync_meta.get("chosen_phase_s"),
            },
        }
        save_bundle(i, out_root, group, target_ts, robot, extra_meta)
        if i % 20 == 0:
            status_put(status_q, f"  saved bundle {i:06d}/{len(groups)}") if status_q is not None else print(
                f"  saved bundle {i:06d}/{len(groups)}"
            )

    dt = time.monotonic() - t0
    summary = {
        "save_duration_s": dt,
        "num_saved_bundles": len(groups),
    }
    if robot_errors:
        arr = np.asarray(robot_errors, dtype=np.float64)
        summary.update(
            {
                "robot_sync_error_abs_ms_min": float(arr.min()),
                "robot_sync_error_abs_ms_p50": float(np.percentile(arr, 50)),
                "robot_sync_error_abs_ms_p90": float(np.percentile(arr, 90)),
                "robot_sync_error_abs_ms_p95": float(np.percentile(arr, 95)),
                "robot_sync_error_abs_ms_max": float(arr.max()),
            }
        )
    status_put(status_q, f"Finished saving bundles in {dt:.2f}s") if status_q is not None else print(
        f"Finished saving bundles in {dt:.2f}s"
    )
    return summary


# -------------------------
# Preview
# -------------------------
def update_preview(store: SharedMemoryStore, scale: float) -> bool:
    latest = store.snapshot_latest()
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
# Child processes
# -------------------------
def _write_shared_command(cmd_seq, cmd_values, cmd_ts, values) -> None:
    """Single-writer seqlock-style command publish."""
    cmd_seq.value += 1  # odd = write in progress
    for i in range(6):
        cmd_values[i] = float(values[i])
    cmd_ts.value = time.monotonic()
    cmd_seq.value += 1  # even = stable snapshot


def _read_shared_command(cmd_seq, cmd_values, cmd_ts):
    for _ in range(8):
        s1 = int(cmd_seq.value)
        if s1 & 1:
            continue
        vals = tuple(float(cmd_values[i]) for i in range(6))
        ts = float(cmd_ts.value)
        s2 = int(cmd_seq.value)
        if s1 == s2 and not (s2 & 1):
            return vals, ts
    return tuple(float(cmd_values[i]) for i in range(6)), float(cmd_ts.value)


def recorder_process_main(
    args: argparse.Namespace,
    record_flag,
    stop_flag,
    recorder_ready_flag,
    robot_states_ready_flag,
    status_q: mp.Queue,
    state_ts,
    state_q,
    state_success,
    state_count,
) -> None:
    preset = RESOLUTION_PRESETS[args.resolution]
    rgb_size = tuple(preset["rgb"])
    mono_size = tuple(preset["mono"])
    source_names = [f"oak_{mxid}" for mxid in args.oak_ids]
    reader_stop_evt = threading.Event()
    robot_states: List[Dict] = []
    start_wall: Optional[str] = None
    out_root: Optional[Path] = None
    recorded = False

    try:
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
                status_q=status_q,
            )
            projectors = {info["source_name"]: DeviceTimeProjector(info["source_name"]) for info in oaks.values()}
            store = SharedMemoryStore(source_names)

            threads = []
            for info in oaks.values():
                name = info["source_name"]
                t = threading.Thread(
                    target=reader_loop,
                    args=(name, info["queue"], projectors[name], store, reader_stop_evt, record_flag),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            status_put(status_q, f"Warming up OAKs for {args.warmup_seconds:.2f}s...")
            warmup_end = time.monotonic() + args.warmup_seconds
            while time.monotonic() < warmup_end and not bool(stop_flag.value):
                if args.preview:
                    if not update_preview(store, args.preview_scale):
                        status_put(status_q, "Preview quit during warmup.")
                        stop_flag.value = 1
                        break
                time.sleep(0.01)

            status_put(status_q, "Freezing OAK device->host receive-offset diagnostics:")
            for name, prj in projectors.items():
                prj.freeze(args.freeze_percentile)
                status_put(status_q, f"  {name}: diagnostic_offset={prj.fixed_offset:.6f}, samples={len(prj.offset_samples)}")

            avail_gib = get_mem_available_gib()
            est_gib = estimate_ram_gib(len(args.oak_ids), rgb_size, args.fps, args.max_duration)
            status_put(status_q, "========== Multiprocess gamepad + OAK recorder ready ==========")
            status_put(status_q, f"Resolution: {args.resolution} RGB {rgb_size} mono/depth {mono_size}")
            status_put(status_q, f"FPS: camera output={args.fps}, sensor={args.sensor_fps}, offline sync={args.sync_fps}")
            status_put(status_q, f"Robot state sampling from FCI cache: {args.robot_state_hz:.1f} Hz")
            status_put(status_q, f"Max duration: {args.max_duration:.2f}s")
            status_put(status_q, f"Estimated RAM for raw OAK frames: ~{est_gib:.1f} GiB")
            if avail_gib is not None:
                status_put(status_q, f"Current MemAvailable: {avail_gib:.1f} GiB")
                if est_gib > 0.75 * avail_gib:
                    status_put(status_q, "WARNING: estimated memory is high compared with available RAM.")
            recorder_ready_flag.value = 1
            status_put(status_q, "Press Square to START recording. Press Square again to STOP and save.")

            while not bool(stop_flag.value) and not bool(record_flag.value):
                if args.preview:
                    if not update_preview(store, args.preview_scale):
                        stop_flag.value = 1
                        break
                time.sleep(0.01)

            if bool(stop_flag.value) and not bool(record_flag.value):
                reader_stop_evt.set()
                for t in threads:
                    t.join(timeout=2.0)
                if args.preview:
                    cv2.destroyAllWindows()
                status_put(status_q, "Recorder stopped before recording started.")
                return

            store.clear_recorded()
            out_root = args.output_root
            if out_root is None:
                out_root = Path.cwd() / f"{SAVE_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                out_root = Path(out_root)
            if out_root.exists() and not args.overwrite:
                existing = list(out_root.glob("bundle_*"))
                if existing or (out_root / "recording_summary.json").exists():
                    raise RuntimeError(
                        f"Output folder {out_root} already contains recording files. Use --overwrite or choose a new folder."
                    )
            ensure_dir(out_root)

            start_wall = datetime.now().isoformat()
            record_start = time.monotonic()
            record_end = record_start + args.max_duration
            last_report = record_start
            recorded = True
            status_put(status_q, f"Start MEMORY recording -> {out_root}")
            status_put(status_q, "Camera capture and robot state sampling run independently; sync is offline.")

            while not bool(stop_flag.value) and bool(record_flag.value):
                if args.preview:
                    if not update_preview(store, args.preview_scale):
                        status_put(status_q, "Preview quit. Stopping recording.")
                        record_flag.value = 0
                        break
                now = time.monotonic()
                if now >= record_end:
                    status_put(status_q, f"Reached max duration {args.max_duration:.2f}s. Stopping recording.")
                    record_flag.value = 0
                    break
                if now - last_report >= 1.0:
                    counts = store.get_record_counts()
                    elapsed = now - record_start
                    mem = get_mem_available_gib()
                    mem_s = f", MemAvailable={mem:.1f}GiB" if mem is not None else ""
                    status_put(
                        status_q,
                        f"  t={elapsed:5.1f}s OAK_counts={counts} robot_states={int(state_count.value)}{mem_s}",
                    )
                    last_report = now
                time.sleep(0.005)

            # Wait briefly for the robot loop to observe the falling edge and publish final state count.
            deadline = time.monotonic() + 2.0
            while not bool(robot_states_ready_flag.value) and time.monotonic() < deadline:
                time.sleep(0.002)

            reader_stop_evt.set()
            for t in threads:
                t.join(timeout=2.0)
            if args.preview:
                cv2.destroyAllWindows()

        if not recorded or out_root is None:
            status_put(status_q, "No recording was captured.")
            return

        frames_by_source = store.get_frames_copy()
        for name in source_names:
            frames_by_source[name].sort(key=lambda f: f.match_ts)

        robot_states = snapshot_robot_states(state_ts, state_q, state_success, state_count)
        robot_states.sort(key=lambda s: float(s["host_ts"]))
        raw_robot_path = save_raw_robot_states_npz(out_root, robot_states)

        status_put(status_q, "Recording stopped. Raw counts:")
        for name in source_names:
            status_put(status_q, f"  {name}: {len(frames_by_source[name])} frames")
        status_put(status_q, f"  robot_states: {len(robot_states)}")
        if raw_robot_path is not None:
            status_put(status_q, f"  raw robot states: {raw_robot_path}")

        status_put(status_q, "Running offline fixed-rate synchronization...")
        groups, target_times, sync_meta = offline_fixed_rate_sync(
            frames_by_source=frames_by_source,
            source_names=source_names,
            sync_fps=args.sync_fps,
            tolerance_ms=args.sync_tolerance_ms,
            phase_steps=args.phase_steps,
            allow_reuse=args.allow_reuse,
        )
        status_put(status_q, "Offline sync result:")
        status_put(status_q, f"  target timesteps : {sync_meta['num_target_timesteps']}")
        status_put(status_q, f"  saved timesteps  : {sync_meta['num_saved_timesteps']}")
        status_put(status_q, f"  dropped timesteps: {sync_meta['num_dropped_timesteps']}")
        status_put(status_q, f"  chosen phase s   : {sync_meta['chosen_phase_s']:.6f}")
        status_put(
            status_q,
            "  match span p50/p95/max ms: "
            f"{sync_meta.get('match_span_ms_p50')} / {sync_meta.get('match_span_ms_p95')} / {sync_meta.get('match_span_ms_max')}",
        )

        args_meta = {
            "robot_ip": args.robot_ip,
            "requested_oak_ids": args.oak_ids,
            "requested_oak_rgb_resolution": list(rgb_size),
            "requested_oak_mono_resolution": list(mono_size),
            "resolution_preset": args.resolution,
            "requested_output_fps": args.fps,
            "requested_sensor_fps": args.sensor_fps,
            "requested_sync_fps": args.sync_fps,
            "requested_robot_state_hz": args.robot_state_hz,
            "requested_max_duration_s": args.max_duration,
            "enable_undistortion": args.enable_undistortion,
            "sync_tolerance_ms": args.sync_tolerance_ms,
            "memory_first_recording": True,
            "offline_sync_mode": "fixed-rate-camera + interpolated-robot-joints",
            "robot_state_source": "direct_1khz_fci_state_shared_buffer_no_get_joints",
            "robot_control_architecture": "072_multiprocess_shared_command_direct_active_control",
        }
        projectors_meta = {name: prj.as_meta() for name, prj in projectors.items()}
        save_summary = save_all_bundles(
            out_root=out_root,
            groups=groups,
            target_times=target_times,
            robot_states=robot_states,
            args_meta=args_meta,
            sync_meta=sync_meta,
            projectors_meta=projectors_meta,
            status_q=status_q,
        )

        summary = {
            "created_at_wallclock": datetime.now().isoformat(),
            "record_start_wallclock": start_wall,
            "depthai_version": dai.__version__,
            "args": vars(args) | {"output_root": str(out_root)},
            "raw_frame_counts": {name: len(frames_by_source[name]) for name in source_names},
            "record_counts": store.record_counts,
            "raw_counts_including_warmup": store.raw_counts,
            "reader_errors": store.errors,
            "projectors": projectors_meta,
            "offline_sync": sync_meta,
            "robot_states": robot_state_stats(robot_states),
            "raw_robot_states_file": str(raw_robot_path) if raw_robot_path is not None else None,
            "save_summary": save_summary,
        }
        with open(out_root / "recording_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        status_put(status_q, "Done saving.")
        status_put(status_q, f"Saved {len(groups)} synchronized bundles to {out_root.resolve()}")
        status_put(status_q, f"Summary: {out_root / 'recording_summary.json'}")

    except KeyboardInterrupt:
        stop_flag.value = 1
    except Exception as e:
        status_put(status_q, f"[recorder] ERROR: {type(e).__name__}: {e}")
        stop_flag.value = 1
        raise
    finally:
        recorder_ready_flag.value = 0
        try:
            reader_stop_evt.set()
            if args.preview:
                cv2.destroyAllWindows()
        except Exception:
            pass
        # One recording per invocation, matching the old recorder behavior.
        stop_flag.value = 1
        status_put(status_q, "Recorder process exiting.")


def gamepad_process_main(
    args: argparse.Namespace,
    record_flag,
    stop_flag,
    recorder_ready_flag,
    robot_ready_flag,
    status_q: mp.Queue,
    cmd_seq,
    cmd_values,
    cmd_ts,
    gripper_toggle_seq,
) -> None:
    import pygame

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        status_put(status_q, "No controller found. Connect a PS controller and try again.")
        stop_flag.value = 1
        pygame.quit()
        return

    joy = pygame.joystick.Joystick(0)
    joy.init()
    status_put(status_q, f"Controller: {joy.get_name()}")
    status_put(status_q, f"  Axes: {joy.get_numaxes()}  Buttons: {joy.get_numbuttons()}")

    def axis(i: int) -> float:
        return joy.get_axis(i) if i < joy.get_numaxes() else 0.0

    def btn(i: int) -> bool:
        return bool(joy.get_button(i)) if i < joy.get_numbuttons() else False

    def trig(i: int) -> float:
        return max(0.0, (axis(i) + 1.0) / 2.0)

    ema_alpha = 0.3
    filtered = [0.0] * 6
    prev_cmd = [0.0] * 6
    max_accel = [4.0] * 3 + [10.0] * 3
    last_filter_t = time.monotonic()
    last_loop_t = time.perf_counter()
    record_btn_prev = False
    circle_prev = False
    cross_prev = False
    dpad_prev = False
    stalls = []

    # This architecture intentionally keeps pygame in its own process.
    try:
        while not bool(robot_ready_flag.value) and not bool(stop_flag.value):
            pygame.event.pump()
            time.sleep(0.01)

        status_put(status_q, "Multiprocess teleop ready.")
        status_put(status_q, "Square=start/stop recording. Circle=quit/save.")
        status_put(status_q, "Cross/home is disabled during MP teleop; D-pad up toggles gripper in a separate process.")

        while not bool(stop_flag.value):
            loop_now = time.perf_counter()
            loop_gap = loop_now - last_loop_t
            last_loop_t = loop_now
            if loop_gap > 0.050:
                stalls.append(loop_gap)
                status_put(status_q, f"[GAMEPAD LOOP STALL] {loop_gap * 1000.0:.1f} ms")

            pygame.event.pump()

            raw = [0.0] * 6
            raw[0] = fr3.deadzone(axis(1), DZ) * MAX_LIN
            raw[1] = fr3.deadzone(axis(0), DZ) * MAX_LIN
            lt = fr3.deadzone(trig(2), DZ)
            raw[2] = lt * MAX_LIN * (1.0 if btn(PS_LB) else -1.0)
            raw[3] = fr3.deadzone(-axis(3), DZ) * MAX_ANG
            raw[4] = fr3.deadzone(axis(4), DZ) * MAX_ANG
            rt = fr3.deadzone(trig(5), DZ)
            raw[5] = rt * MAX_ANG * (1.0 if btn(PS_RB) else -1.0)

            # Input filter adapted from fr3's Gamepad.update (third_party/fr3/src/fr3/_gamepad.py).
            for i in range(6):
                alpha = min(ema_alpha * 2.5, 0.7) if abs(raw[i]) < abs(filtered[i]) else ema_alpha
                filtered[i] = alpha * raw[i] + (1.0 - alpha) * filtered[i]
            if all(abs(r) < 1e-10 for r in raw):
                for i in range(6):
                    if abs(filtered[i]) < 5e-4:
                        filtered[i] = 0.0

            now = time.monotonic()
            dt = min(now - last_filter_t, 0.1)
            last_filter_t = now
            out = list(filtered)
            for i in range(6):
                delta = out[i] - prev_cmd[i]
                max_delta = max_accel[i] * dt
                if abs(delta) > max_delta:
                    out[i] = prev_cmd[i] + math.copysign(max_delta, delta)

            lin_mag = math.sqrt(sum(out[i] * out[i] for i in range(3)))
            if lin_mag > MAX_LIN:
                scale = MAX_LIN / lin_mag
                for i in range(3):
                    out[i] *= scale
            ang_mag = math.sqrt(sum(out[i] * out[i] for i in range(3, 6)))
            if ang_mag > MAX_ANG:
                scale = MAX_ANG / ang_mag
                for i in range(3, 6):
                    out[i] *= scale

            prev_cmd[:] = out
            _write_shared_command(cmd_seq, cmd_values, cmd_ts, out)

            circle_now = btn(PS_CIRCLE)
            if circle_now and not circle_prev:
                if bool(record_flag.value):
                    status_put(status_q, "Circle pressed -> stop recording and save current data.")
                    record_flag.value = 0
                    time.sleep(0.05)
                stop_flag.value = 1
                break
            circle_prev = circle_now

            record_btn = btn(PS_SQUARE)
            if record_btn and not record_btn_prev:
                if not bool(recorder_ready_flag.value):
                    status_put(status_q, "Square ignored: OAK recorder is not ready yet.")
                elif bool(record_flag.value):
                    record_flag.value = 0
                    status_put(status_q, "Record toggle -> OFF. Offline sync/save will start.")
                else:
                    record_flag.value = 1
                    status_put(status_q, "Record toggle -> ON. Memory recording started.")
            record_btn_prev = record_btn

            # Avoid silently changing robot control sessions while recording.
            cross_now = btn(PS_CROSS)
            if cross_now and not cross_prev:
                status_put(status_q, "Cross/home disabled in MP mode. Restart with --home-on-start True if needed.")
            cross_prev = cross_now

            dpad_up = joy.get_numhats() > 0 and joy.get_hat(0)[1] == 1
            if dpad_up and not dpad_prev:
                gripper_toggle_seq.value += 1
                status_put(status_q, "D-pad up -> gripper toggle requested.")
            dpad_prev = dpad_up

            time.sleep(0.01)  # ~100 Hz producer

    except KeyboardInterrupt:
        stop_flag.value = 1
    except Exception as e:
        status_put(status_q, f"[gamepad] ERROR: {type(e).__name__}: {e}")
        stop_flag.value = 1
        raise
    finally:
        try:
            _write_shared_command(cmd_seq, cmd_values, cmd_ts, [0.0] * 6)
        except Exception:
            pass
        pygame.quit()
        status_put(status_q, f"Gamepad stalls >50ms: {len(stalls)}")
        if stalls:
            status_put(status_q, f"Gamepad max stall: {max(stalls) * 1000.0:.1f} ms")
        status_put(status_q, "Gamepad process exiting.")



def gripper_process_main(
    args: argparse.Namespace,
    stop_flag,
    robot_ready_flag,
    status_q: mp.Queue,
    gripper_toggle_seq,
) -> None:
    """Own blocking gripper calls in a separate process."""
    from fr3._gripper import GripperControl

    last_seq = int(gripper_toggle_seq.value)
    gripper = None
    is_open = False  # fallback: first toggle opens

    try:
        # Wait until optional arm homing is finished and the arm-only FCI
        # connection is active. This avoids overlapping gripper TCP clients.
        while not bool(robot_ready_flag.value) and not bool(stop_flag.value):
            time.sleep(0.01)

        if bool(stop_flag.value):
            return

        gripper = GripperControl(args.robot_ip)

        try:
            st = gripper.read_state()
            width = float(st.get("width", 0.0))
            is_grasped = bool(st.get("is_grasped", False))
            is_open = (width >= 0.06) and not is_grasped
            status_put(
                status_q,
                f"Gripper ready: width={width:.4f} m, is_grasped={is_grasped}, "
                f"state={'OPEN' if is_open else 'CLOSED/PARTIAL'}",
            )
        except Exception as e:
            status_put(
                status_q,
                f"Gripper connected; initial state read failed "
                f"({type(e).__name__}: {e}). First toggle will OPEN.",
            )
            is_open = False

        while not bool(stop_flag.value):
            current_seq = int(gripper_toggle_seq.value)

            if current_seq != last_seq:
                last_seq += 1

                if is_open:
                    status_put(status_q, "Gripper action -> GRASP/CLOSE...")
                    ok = gripper.grasp(
                        width=GRIPPER_GRASP_WIDTH,
                        speed=GRIPPER_SPEED,
                        force=GRIPPER_GRASP_FORCE,
                        epsilon_inner=GRIPPER_GRASP_EPSILON,
                        epsilon_outer=GRIPPER_GRASP_EPSILON,
                    )
                    status_put(status_q, f"Gripper grasp result: {ok}")
                    is_open = False
                else:
                    status_put(status_q, "Gripper action -> OPEN...")
                    ok = gripper.open(speed=GRIPPER_SPEED)
                    status_put(status_q, f"Gripper open result: {ok}")
                    is_open = True

                try:
                    st = gripper.read_state()
                    width = float(st.get("width", 0.0))
                    is_grasped = bool(st.get("is_grasped", False))
                    is_open = (width >= 0.06) and not is_grasped
                    status_put(
                        status_q,
                        f"Gripper state: width={width:.4f} m, is_grasped={is_grasped}",
                    )
                except Exception:
                    pass

            time.sleep(0.01)

    except KeyboardInterrupt:
        stop_flag.value = 1
    except Exception as e:
        status_put(status_q, f"[gripper] ERROR: {type(e).__name__}: {e}")
        stop_flag.value = 1
        raise
    finally:
        if gripper is not None:
            try:
                gripper.stop()
            except Exception:
                pass
        status_put(status_q, "Gripper process exiting.")

def robot_control_process_main(
    args: argparse.Namespace,
    record_flag,
    stop_flag,
    robot_ready_flag,
    robot_states_ready_flag,
    status_q: mp.Queue,
    cmd_seq,
    cmd_values,
    cmd_ts,
    state_ts,
    state_q,
    state_success,
    state_count,
    state_capacity: int,
) -> None:
    # Keep BLAS helper pools out of the realtime process.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    from pylibfranka import CartesianPose, ControllerMode, Robot

    fr3.ROBOT_IP = args.robot_ip
    DT = 0.001
    K = 15.0  # smoother constants and loop adapted from fr3's ControlThread._velocity_loop (third_party/fr3/src/fr3/_control.py)
    SNAP_THRESHOLD = 1e-6
    max_accel = [6.0] * 3 + [12.0] * 3
    max_jerk = [0.04] * 3 + [0.08] * 3
    sample_period = 1.0 / float(args.robot_state_hz)

    def clip(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    ticks = 0
    min_success = 1.0
    max_compute_us = 0.0
    compute_over_500us = 0
    stale_command_ticks = 0
    state_overflow = False
    was_recording = False
    next_state_sample_ts: Optional[float] = None
    error_text: Optional[str] = None
    robot = None

    try:
        # fr3.FR3() creates BOTH arm and gripper connections. Use it only
        # transiently for the existing high-level home routine, if requested.
        if args.home_on_start:
            status_put(status_q, "Robot homing before multiprocess teleop...")
            with fr3.FR3(ip=args.robot_ip) as home_wrapper:
                home_wrapper.home(blocking=True)

        # Arm-only connection. This is the key fix: do not keep an FR3 wrapper
        # alive while the dedicated gripper process owns port 1338.
        robot = Robot(args.robot_ip)
        robot.automatic_error_recovery()
        robot.set_collision_behavior(
            [20.0] * 7,
            [40.0] * 7,
            [10.0] * 6,
            [20.0] * 6,
        )
        robot.set_joint_impedance(
            [3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0]
        )
        robot.set_cartesian_impedance(
            [3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0]
        )

        active_control = robot.start_cartesian_pose_control(
            ControllerMode.CartesianImpedance
        )

        smoothed = [0.0] * 6
        accel = [0.0] * 6
        target_4x4 = None
        anchor_pos = None
        robot_ready_flag.value = 1
        status_put(
            status_q,
            "Robot direct 1 kHz multiprocess control active "
            "(arm-only connection; gripper is separate).",
        )

        while True:
            state, _duration = active_control.readOnce()
            state_host_ts = time.monotonic()
            compute_t0 = time.perf_counter_ns()

            recording_now = bool(record_flag.value)
            if recording_now and not was_recording:
                reset_robot_state_buffer(state_count)
                robot_states_ready_flag.value = 0
                next_state_sample_ts = state_host_ts
                status_put(status_q, "Robot state capture -> ON (from FCI stream, no get_joints).")

            if recording_now:
                if next_state_sample_ts is None:
                    next_state_sample_ts = state_host_ts
                if state_host_ts + 1e-12 >= next_state_sample_ts:
                    success_rate = float(getattr(state, "control_command_success_rate", 1.0))
                    ok = append_robot_state_shared(
                        state_ts=state_ts,
                        state_q=state_q,
                        state_success=state_success,
                        state_count=state_count,
                        capacity=state_capacity,
                        host_ts=state_host_ts,
                        joints=state.q,
                        success_rate=success_rate,
                    )
                    if not ok:
                        state_overflow = True
                    # Keep a fixed sampling grid without burst catch-up writes.
                    while next_state_sample_ts <= state_host_ts:
                        next_state_sample_ts += sample_period

            if was_recording and not recording_now:
                # Current count is now stable for the recorder process.
                robot_states_ready_flag.value = 1
                status_put(status_q, f"Robot state capture -> OFF ({int(state_count.value)} samples).")
            was_recording = recording_now

            if target_4x4 is None:
                target_4x4 = np.array(state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F").copy()

            stopping = bool(stop_flag.value)
            if stopping:
                target = (0.0,) * 6
            else:
                target, latest_cmd_ts = _read_shared_command(cmd_seq, cmd_values, cmd_ts)
                if time.monotonic() - latest_cmd_ts > args.command_timeout:
                    target = (0.0,) * 6
                    stale_command_ticks += 1

            for i in range(6):
                err = target[i] - smoothed[i]
                decelerating = abs(target[i]) < abs(smoothed[i])
                jerk_i = max_jerk[i] * (3.0 if decelerating else 1.0)
                if abs(accel[i]) > 1e-10:
                    stop_vel_change = accel[i] * abs(accel[i]) / jerk_i * DT * 0.5
                else:
                    stop_vel_change = 0.0
                predicted = smoothed[i] + stop_vel_change
                needs_brake = (
                    (accel[i] > 0.0 and predicted >= target[i])
                    or (accel[i] < 0.0 and predicted <= target[i])
                )
                desired = 0.0 if needs_brake else clip(K * err, -max_accel[i], max_accel[i])
                da = clip(desired - accel[i], -jerk_i, jerk_i)
                accel[i] += da
                smoothed[i] += accel[i] * DT
                if abs(smoothed[i] - target[i]) < SNAP_THRESHOLD and abs(accel[i]) < SNAP_THRESHOLD:
                    smoothed[i] = target[i]
                    accel[i] = 0.0

            target_lin_zero = max(abs(target[i]) for i in range(3)) < 1e-10
            smoothed_lin_near_zero = max(abs(smoothed[i]) for i in range(3)) < 1e-5
            if target_lin_zero and smoothed_lin_near_zero:
                if anchor_pos is None:
                    anchor_pos = target_4x4[:3, 3].copy()
                target_4x4[:3, 3] = anchor_pos
            else:
                if not target_lin_zero:
                    anchor_pos = None
                target_4x4[0, 3] += smoothed[0] * DT
                target_4x4[1, 3] += smoothed[1] * DT
                target_4x4[2, 3] += smoothed[2] * DT

            wx, wy, wz = smoothed[3], smoothed[4], smoothed[5]
            omega_norm = math.sqrt(wx * wx + wy * wy + wz * wz)
            angle = omega_norm * DT
            if angle > 1e-10 and omega_norm > 0.0:
                ax, ay, az = wx / omega_norm, wy / omega_norm, wz / omega_norm
                s, c = math.sin(angle), math.cos(angle)
                C = 1.0 - c
                dR = np.array(
                    [
                        [c + ax * ax * C, ax * ay * C - az * s, ax * az * C + ay * s],
                        [ay * ax * C + az * s, c + ay * ay * C, ay * az * C - ax * s],
                        [az * ax * C - ay * s, az * ay * C + ax * s, c + az * az * C],
                    ],
                    dtype=np.float64,
                )
                target_4x4[:3, :3] = dR @ target_4x4[:3, :3]

            ticks += 1
            # Re-orthonormalize every 5000 ticks (5 s) rather than fr3's 500 ticks (0.5 s); this ran stably in teleop tests.
            if ticks % 5000 == 0:
                U, _, Vt = np.linalg.svd(target_4x4[:3, :3])
                target_4x4[:3, :3] = U @ Vt

            success_rate = float(getattr(state, "control_command_success_rate", 1.0))
            min_success = min(min_success, success_rate)

            flat = list(target_4x4.flatten(order="F"))
            cmd = CartesianPose(flat)
            fully_stopped = (
                max(abs(v) for v in smoothed) < 1e-6
                and max(abs(a) for a in accel) < 1e-6
            )
            if stopping and fully_stopped:
                cmd.motion_finished = True

            compute_us = (time.perf_counter_ns() - compute_t0) / 1000.0
            max_compute_us = max(max_compute_us, compute_us)
            if compute_us > 500.0:
                compute_over_500us += 1

            active_control.writeOnce(cmd)

            if stopping and fully_stopped:
                break

        try:
            robot.stop()
        except Exception:
            pass

    except KeyboardInterrupt:
        stop_flag.value = 1
    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        status_put(status_q, f"[robot] ERROR: {error_text}")
        stop_flag.value = 1
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if was_recording or int(state_count.value) > 0:
            robot_states_ready_flag.value = 1
        robot_ready_flag.value = 0
        status_put(status_q, "===== Robot control diagnostics =====")
        status_put(status_q, f"  ticks: {ticks}")
        status_put(status_q, f"  min control_command_success_rate: {min_success:.4f}")
        status_put(status_q, f"  max Python compute: {max_compute_us:.1f} us")
        status_put(status_q, f"  compute >500us: {compute_over_500us}")
        status_put(status_q, f"  stale command ticks: {stale_command_ticks}")
        status_put(status_q, f"  robot state samples: {int(state_count.value)}")
        status_put(status_q, f"  robot state buffer overflow: {state_overflow}")
        if error_text:
            status_put(status_q, f"  error: {error_text}")
        status_put(status_q, "Robot control process exiting.")


# -------------------------
# CLI / main process
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multiprocess PS-gamepad teleop + memory-first 3-OAK RGB-D + synchronized FR3 states."
    )
    p.add_argument("robot_ip", nargs="?", default=DEFAULT_ROBOT_IP, help="FR3 FCI IP address; defaults to $AURA_ROBOT_IP.")
    p.add_argument("--oak-ids", nargs="+", default=DEFAULT_OAK_IDS, help="OAK MXIDs; defaults to $AURA_OAK_IDS, camera-index order.")
    p.add_argument("--resolution", choices=sorted(RESOLUTION_PRESETS.keys()), default="low")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help="DepthAI output FPS.")
    p.add_argument("--sensor-fps", type=int, default=DEFAULT_SENSOR_FPS, help="OAK sensor FPS.")
    p.add_argument("--sync-fps", type=float, default=DEFAULT_SYNC_FPS, help="Offline fixed-rate output FPS.")
    p.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION_S, help="Max memory recording duration in seconds.")
    p.add_argument("--output-root", type=Path, default=None, help="Output folder. Default: ./recording_sync_robot_YYYYMMDD_HHMMSS in the current working directory.")
    p.add_argument("--warmup-seconds", type=float, default=DEFAULT_WARMUP_SECONDS)
    p.add_argument("--preview", type=str2bool, default=False)
    p.add_argument("--preview-scale", type=float, default=PREVIEW_SCALE)
    p.add_argument("--sync-tolerance-ms", type=float, default=DEFAULT_SYNC_TOLERANCE_MS)
    p.add_argument("--oak-internal-sync-ms", type=float, default=DEFAULT_INTERNAL_SYNC_MS)
    p.add_argument("--enable-undistortion", type=str2bool, default=True)
    p.add_argument("--freeze-percentile", type=float, default=DEFAULT_OFFSET_FREEZE_PERCENTILE)
    p.add_argument("--phase-steps", type=int, default=DEFAULT_PHASE_STEPS)
    p.add_argument("--allow-reuse", type=str2bool, default=False, help="Allow reusing camera frames for adjacent timesteps. Usually False for dynamics.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--robot-state-hz",
        type=float,
        default=STATE_PUBLISH_HZ,
        help="Sample rate from the already-received FCI state. Default 100 Hz; this does NOT call get_joints().",
    )
    p.add_argument(
        "--command-timeout",
        type=float,
        default=0.5,
        help="Safety watchdog for stale gamepad commands. Keep at 0.5 s unless you have a specific reason.",
    )
    p.add_argument(
        "--home-on-start",
        type=str2bool,
        default=True,
        help="Home the robot once before starting direct teleop. Default True: the robot moves to its home pose right after launch, without a prompt.",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_duration <= 0:
        raise ValueError("--max-duration must be positive")
    if args.max_duration > HARD_MAX_DURATION_S:
        raise ValueError(
            f"For safety, --max-duration is capped at {HARD_MAX_DURATION_S:.1f}s. "
            f"You requested {args.max_duration:.1f}s."
        )
    if args.robot_state_hz <= 0 or args.robot_state_hz > 1000:
        raise ValueError("--robot-state-hz must be in (0, 1000]")
    if args.sync_fps <= 0:
        raise ValueError("--sync-fps must be positive")
    if args.command_timeout <= 0:
        raise ValueError("--command-timeout must be positive")
    if not args.robot_ip:
        raise ValueError("No robot IP: pass it as the first argument or set AURA_ROBOT_IP in config/aura.env and load it (set -a; source config/aura.env; set +a).")
    if not args.oak_ids:
        raise ValueError(
            "No OAK cameras: set AURA_OAK_IDS in config/aura.env (space-separated MXIDs, camera-index order) "
            "and load it (set -a; source config/aura.env; set +a) or pass --oak-ids."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    ctx = mp.get_context("spawn")

    # Simple raw flags avoid putting multiprocessing.Event locks in the 1 kHz loop.
    record_flag = ctx.RawValue(ctypes.c_int, 0)
    stop_flag = ctx.RawValue(ctypes.c_int, 0)
    recorder_ready_flag = ctx.RawValue(ctypes.c_int, 0)
    robot_ready_flag = ctx.RawValue(ctypes.c_int, 0)
    robot_states_ready_flag = ctx.RawValue(ctypes.c_int, 0)

    # Latest gamepad command, seqlock-style.
    cmd_seq = ctx.RawValue(ctypes.c_ulonglong, 0)
    cmd_values = ctx.RawArray(ctypes.c_double, 6)
    cmd_ts = ctx.RawValue(ctypes.c_double, time.monotonic())

    # Rising-edge count for D-pad-up gripper toggles.
    gripper_toggle_seq = ctx.RawValue(ctypes.c_ulonglong, 0)

    # Fixed-capacity robot-state buffer. At 100 Hz x 60 s this is tiny (~0.5 MB for q/timestamps).
    capacity = robot_state_capacity(args.max_duration, args.robot_state_hz)
    state_ts = ctx.RawArray(ctypes.c_double, capacity)
    state_q = ctx.RawArray(ctypes.c_double, capacity * 7)
    state_success = ctx.RawArray(ctypes.c_double, capacity)
    state_count = ctx.RawValue(ctypes.c_int, 0)

    status_q: mp.Queue = ctx.Queue()

    recorder_proc = ctx.Process(
        target=recorder_process_main,
        args=(
            args,
            record_flag,
            stop_flag,
            recorder_ready_flag,
            robot_states_ready_flag,
            status_q,
            state_ts,
            state_q,
            state_success,
            state_count,
        ),
        name="recorder_process",
    )
    robot_proc = ctx.Process(
        target=robot_control_process_main,
        args=(
            args,
            record_flag,
            stop_flag,
            robot_ready_flag,
            robot_states_ready_flag,
            status_q,
            cmd_seq,
            cmd_values,
            cmd_ts,
            state_ts,
            state_q,
            state_success,
            state_count,
            capacity,
        ),
        name="robot_1khz_process",
    )
    gamepad_proc = ctx.Process(
        target=gamepad_process_main,
        args=(
            args,
            record_flag,
            stop_flag,
            recorder_ready_flag,
            robot_ready_flag,
            status_q,
            cmd_seq,
            cmd_values,
            cmd_ts,
            gripper_toggle_seq,
        ),
        name="gamepad_100hz_process",
    )
    gripper_proc = ctx.Process(
        target=gripper_process_main,
        args=(
            args,
            stop_flag,
            robot_ready_flag,
            status_q,
            gripper_toggle_seq,
        ),
        name="gripper_process",
    )

    recorder_proc.start()
    robot_proc.start()
    gamepad_proc.start()
    gripper_proc.start()

    procs = [recorder_proc, robot_proc, gamepad_proc, gripper_proc]

    try:
        while True:
            try:
                msg = status_q.get(timeout=MAIN_STATUS_POLL_TIMEOUT_S)
                print(msg, flush=True)
            except queue.Empty:
                pass

            # Any unexpected child exit stops the whole session.
            bad = [p for p in procs if p.exitcode not in (None, 0)]
            if bad:
                stop_flag.value = 1

            if all(not p.is_alive() for p in procs):
                break

    except KeyboardInterrupt:
        stop_flag.value = 1
        record_flag.value = 0
    finally:
        stop_flag.value = 1
        record_flag.value = 0

        # Recorder may need time to close OAKs and save the memory-first session.
        recorder_proc.join(timeout=30.0)
        gamepad_proc.join(timeout=5.0)
        gripper_proc.join(timeout=5.0)
        robot_proc.join(timeout=8.0)

        for p in (gamepad_proc, gripper_proc, robot_proc, recorder_proc):
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

        while True:
            try:
                msg = status_q.get_nowait()
                print(msg, flush=True)
            except queue.Empty:
                break
            except Exception:
                break

        print(
            "Exit codes: "
            f"recorder={recorder_proc.exitcode}, "
            f"robot={robot_proc.exitcode}, "
            f"gamepad={gamepad_proc.exitcode}, "
            f"gripper={gripper_proc.exitcode}",
            flush=True,
        )


if __name__ == "__main__":
    main()
