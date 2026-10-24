#!/usr/bin/env python3
r"""
AnyGrasp Franka execution + memory-first synchronized 3-OAK recording with
parallel robot/recorder execution and per-camera reader threads.

Direct-FCI version: the robot process owns an arm-only pylibfranka Robot and
runs hover/descend/lift with direct 1 kHz CartesianPose motion. PyRoki dense IK
is used only as an offline robustness filter; its joint waypoints are not sent
to the robot. Measured robot states are sampled from the already-received FCI
RobotState (no get_joints() polling during motion). The recorder process owns
the three OAK cameras and runs one reader thread per camera.

Process split
-------------
1) Robot process
   - Owns the arm-only FCI connection and a separate gripper connection
   - Executes hover/descend/lift with a direct 1 kHz CartesianPose loop
   - Aborts before a measured joint enters the configured runtime margin
   - Samples measured q / EE pose directly from FCI state at --state-publish-hz

2) Recorder process
   - Owns 3x OAK capture
   - Records all OAK RGB-D frames into CPU RAM during lift
   - Reads latest measured FR3 pose / joints from shared memory into RAM
   - Runs offline fixed-rate synchronization after recording stops
   - Writes synchronized bundles with the same bundle_*/oak_*/rgb.jpg + depth.npy + meta.json layout

Robot connection model
----------------------
The arm is owned by one low-level pylibfranka Robot connection. The gripper
uses its normal independent connection. The recorder never opens a robot
connection; it only reads the measured state published by the robot process.

Usage
-----
Normally run by scripts/run_anygrasp_to_lift_one_shot.sh (stage 02). By hand,
from the repository root with config/aura.env loaded (set -a; source config/aura.env; set +a):

uv run --project "$AURA_ROBOT_PROJECT" python aura/robot/execute_anygrasp_franka_record_lift.py \
  --calib "$AURA_CALIB_ROOT/robot_calibration_results/robot_calib_result.npz" \
  --grasp-json "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp_results/grasps_table_ik_safe.json" \
  --rank 0 --pose-frame world --orientation-mode anygrasp --rot-offset-rpy 0 1.57079632679 0 \
  --record-output-root "$AURA_DATA_ROOT/franka_lift/<case>/phystwin/raw_recording" \
  --descend-speed 0.02 --home --execute

- Without --execute this is a dry run: it checks the calibration, the pose
  mapping and the PyRoki hover/robustness metadata of the selected grasp, then
  exits without connecting to the robot or opening a camera.
- --ip defaults to $AURA_ROBOT_IP and --record-oak-ids to $AURA_OAK_IDS (MXIDs in
  camera-index order); --execute stops with an error if either is empty.
- The intrinsics in recording_summary.json (calibration_manifest) are read from
  $AURA_CALIB_ROOT/intrinsic_results/<WxH>/oak_<MXID>.npz (default <repo>/calibration);
  if that folder has no files the manifest is empty.
- --grasp-json must be grasps_table_ik_safe.json from
  aura/grasping/filter_grasps_table_collision_pyroki.py and --rank is the index into
  that list (not the AnyGrasp rank). The z offsets, --rot-offset-rpy and the calibration
  must be the filter's; only the hover pose is cross-checked (1 mm / 0.5 deg).
- --motion-speed / --descend-speed / --lift-speed are m/s here. In execute_anygrasp_franka.py
  --motion-speed (hover and lift) and --descend-speed are fr3 speed fractions (no --lift-speed).
- Always pass --record-output-root (the inferred default assumes a flat data/<case>/ layout), and
  a folder without a recording: a used folder is detected only when the lift starts, after the grasp.
- Accepted for compatibility but without effect here: --workspace-*, --allow-outside-workspace,
  --disable-workspace-check, --joint4-*, --disable-joint4-check, --record-warmup-before-lift and
  the --servo-* flags other than --servo-max-angular (see docs/troubleshooting.md, Known limitations).
- --hover-only hangs here (the recorder keeps waiting for a lift): end the run with Ctrl-C; the
  arm stays at the hover pose. The "gripper reports no object" abort hangs the same way, with the
  arm at the grasp pose and the gripper open. For hover tests use execute_anygrasp_franka.py
  --execute --home --hover-only (docs/usage.md, Dry run and hover-only test).
"""

from __future__ import annotations

import argparse
import contextlib
import filecmp
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
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
    raise RuntimeError(
        "Failed to import depthai. Activate the environment where DepthAI v3 is installed."
    ) from e

import fr3
import execute_anygrasp_franka as grasp_exec


# =========================
# User configuration
# =========================
ROBOT_IP = os.environ.get("AURA_ROBOT_IP")  # config/aura.env; the --ip default reads the same variable

FPS = 15
SYNC_TOLERANCE_MS = 45
OAK_INTERNAL_SYNC_MS = 25
WARMUP_SECONDS = 0.0  # default of --record-warmup-before-lift, which has no effect (kept for CLI compatibility)
CAMERA_WARMUP_SECONDS = 2.0  # drain camera streams before robot motion starts

# Capture size (re-set by apply_record_resolution) and per-camera intrinsics for the
# recording manifest. apply_record_resolution() fills the dict from
# $AURA_CALIB_ROOT/intrinsic_results/<WxH>/oak_<MXID>.npz; rig-specific intrinsics
# are not shipped, so it starts empty.
OAK_W = 640
OAK_H = 400

HARDCODED_CAMERA_CONFIGS: Dict[str, dict] = {}

SAVE_PREFIX = "recording_sync_robot"

# OAK MXIDs in camera-index order, from AURA_OAK_IDS (config/aura.env).
TARGET_OAK_IDS: List[str] = os.environ.get("AURA_OAK_IDS", "").split()

RGB_JPEG_QUALITY = 85
PREVIEW_SCALE = 0.5


RECORD_RESOLUTION = "low"


def infer_case_phystwin_output_root(grasp_json: Path) -> Path:
    """Infer data/<case>/phystwin from a grasp JSON stored under the case."""
    grasp_path = Path(grasp_json).expanduser().resolve()
    for candidate in grasp_path.parents:
        if candidate.parent.name == "data":
            return (candidate / "phystwin").resolve()

    raise ValueError(
        "Could not infer data/<case>/phystwin from --grasp-json "
        f"{grasp_path}. Pass --record-output-root explicitly, for example "
        "$AURA_DATA_ROOT/franka_lift/<case>/phystwin/raw_recording (the folder the lift one-shot uses)."
    )


def _scaled_camera_config(cfg: dict, new_w: int, new_h: int, source_label: str) -> dict:
    """Fallback only: scale a pinhole K when a real calibration file is unavailable."""
    old_w = float(cfg["image_width"])
    old_h = float(cfg["image_height"])
    sx = float(new_w) / old_w
    sy = float(new_h) / old_h
    out = dict(cfg)
    out["image_width"] = int(new_w)
    out["image_height"] = int(new_h)
    if cfg.get("K") is not None:
        K = np.asarray(cfg["K"], dtype=float).copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        out["K"] = K.tolist()
    out["config_source"] = source_label
    out["calibration_warning"] = (
        "Real intrinsic calibration file was not found; this K was scaled from low resolution."
    )
    return out


def _repo_root() -> Path:
    """Repository root: $AURA_REPO_ROOT, else two levels above aura/robot/."""
    return Path(os.environ.get("AURA_REPO_ROOT") or Path(__file__).resolve().parents[2])


def _load_real_intrinsics(width: int, height: int) -> Dict[str, dict]:
    """Load the calibrated intrinsics from $AURA_CALIB_ROOT/intrinsic_results/<WxH>/ (default <repo>/calibration)."""
    calib_root = Path(os.environ.get("AURA_CALIB_ROOT") or _repo_root() / "calibration")
    intr_dir = calib_root / "intrinsic_results" / f"{width}x{height}"
    configs: Dict[str, dict] = {}
    if not intr_dir.is_dir():
        return configs

    for npz_path in sorted(intr_dir.glob("oak_*.npz")):
        try:
            d = np.load(npz_path)
            K = np.asarray(d["K"], dtype=float)
            dist = np.asarray(d["dist"], dtype=float).reshape(-1)
            image_size = np.asarray(d["image_size"] if "image_size" in d.files else [width, height], dtype=int).reshape(-1)
            rms = float(np.asarray(d["rms"]).reshape(-1)[0]) if "rms" in d.files else None
            configs[npz_path.stem] = {
                "image_width": int(image_size[0]) if image_size.size >= 1 else int(width),
                "image_height": int(image_size[1]) if image_size.size >= 2 else int(height),
                "K": K.tolist(),
                "dist": dist.tolist(),
                "rms_reprojection_error": rms,
                "config_source": str(npz_path),
                "calibration_warning": None,
            }
        except Exception:
            continue
    return configs


def apply_record_resolution(resolution: str) -> None:
    """Configure capture resolution and load the matching real intrinsic files."""
    global RECORD_RESOLUTION, OAK_W, OAK_H, HARDCODED_CAMERA_CONFIGS

    resolution = str(resolution).lower().strip()
    if resolution not in {"low", "high"}:
        raise ValueError(f"Unknown --record-resolution: {resolution}")

    RECORD_RESOLUTION = resolution
    if resolution == "low":
        OAK_W, OAK_H = 640, 400
    else:
        OAK_W, OAK_H = 1280, 720

    real = _load_real_intrinsics(OAK_W, OAK_H)
    if real:
        HARDCODED_CAMERA_CONFIGS = real
    elif resolution == "low":
        for _name, cfg in HARDCODED_CAMERA_CONFIGS.items():
            cfg["config_source"] = "hardcoded_low_resolution_fallback"
    else:
        HARDCODED_CAMERA_CONFIGS = {
            source_name: _scaled_camera_config(
                cfg, OAK_W, OAK_H, "scaled_from_low_resolution_fallback"
            )
            for source_name, cfg in HARDCODED_CAMERA_CONFIGS.items()
        }


PROJECTOR_EMA_ALPHA = 0.02

OAK_QUEUE_MAXSIZE = 8

MAX_LIN = 0.25
MAX_ANG = 0.4

MAIN_STATUS_POLL_TIMEOUT_S = 0.25
STATE_PUBLISH_HZ = 100.0


# =========================
# Helpers
# =========================

def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")

def td_to_sec(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return x.total_seconds()
    except Exception:
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_robot_calibration_to_phystwin(
    calib_path: Path,
    phystwin_root: Path,
) -> Dict[str, object]:
    """Copy the run's robot calibration to a stable file in phystwin_root."""
    source = Path(calib_path).expanduser().resolve()
    root = Path(phystwin_root).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"robot calibration file not found: {source}")

    ensure_dir(root)
    suffix = source.suffix.lower()
    copied = root / f"robot_calib_result{suffix}"

    if copied.exists():
        if not filecmp.cmp(source, copied, shallow=False):
            raise RuntimeError(
                f"{copied} already exists but differs from --calib {source}. "
                "Refusing to overwrite the calibration belonging to existing "
                "case data. Use a new case/output folder, or remove the old "
                "recording and calibration intentionally."
            )
    else:
        temp = root / f".{copied.name}.tmp"
        try:
            shutil.copy2(source, temp)
            if not filecmp.cmp(source, temp, shallow=False):
                raise RuntimeError(
                    f"Robot calibration copy verification failed: {source} -> {temp}"
                )
            temp.replace(copied)
        finally:
            if temp.exists():
                temp.unlink()

    return {
        "source_path_at_copy_time": str(source),
        "path_relative_to_phystwin": str(copied.relative_to(root)),
        "size_bytes": int(copied.stat().st_size),
        "copied_or_reused_at_wallclock": datetime.now().isoformat(),
    }


def build_calibration_manifest() -> Dict[str, dict]:
    manifest: Dict[str, dict] = {}
    for source_name, cfg in HARDCODED_CAMERA_CONFIGS.items():
        manifest[source_name] = {
            "record_resolution": RECORD_RESOLUTION,
            "config_source": cfg.get("config_source", "hardcoded_in_script"),
            "calibration_warning": cfg.get("calibration_warning"),
            "image_width": int(cfg["image_width"]),
            "image_height": int(cfg["image_height"]),
            "K": cfg.get("K"),
            "dist": cfg.get("dist"),
            "rms_reprojection_error": cfg.get("rms_reprojection_error"),
        }
    return manifest


def maybe_resize(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def status_put(status_q: mp.Queue, msg: str) -> None:
    try:
        status_q.put_nowait(msg)
    except Exception:
        pass


def create_shared_robot_state(ctx: mp.context.BaseContext) -> Dict[str, object]:
    return {
        "lock": ctx.Lock(),
        "valid": ctx.Value("i", 0),
        "host_ts": ctx.Value("d", 0.0),
        "pose": ctx.Array("d", 6),
        "joints": ctx.Array("d", 7),
    }


def write_shared_robot_state(shared: Dict[str, object], state: Dict) -> None:
    pose_vals = state.get("pose_xyzrpy", [0.0] * 6)
    joint_vals = state.get("joint_angles_rad", [0.0] * 7)
    with shared["lock"]:
        shared["host_ts"].value = float(state.get("host_ts", time.monotonic()))
        for i in range(6):
            shared["pose"][i] = float(pose_vals[i])
        for i in range(7):
            shared["joints"][i] = float(joint_vals[i])
        shared["valid"].value = 1


def read_shared_robot_state(shared: Dict[str, object]) -> Dict:
    with shared["lock"]:
        valid = int(shared["valid"].value)
        host_ts = float(shared["host_ts"].value)
        pose = [float(shared["pose"][i]) for i in range(6)]
        joints = [float(shared["joints"][i]) for i in range(7)]
    if not valid:
        return {
            "error": "No shared robot state has been published yet.",
            "host_ts": time.monotonic(),
        }
    return {
        "host_ts": host_ts,
        "pose_xyzrpy": pose,
        "joint_angles_rad": joints,
        "state_source": "actual_measured_shared_state",
    }


# =========================
# Memory-first 3-OAK recorder
# =========================

DEFAULT_OFFSET_FREEZE_PERCENTILE = 5.0
HARD_MEMORY_MAX_DURATION_S = 120.0


@dataclass
class MemFrame:
    source: str
    host_ts: float
    device_ts: Optional[float]
    match_ts: float
    frame_number: Optional[int]
    rgb: np.ndarray
    depth: np.ndarray


class MemoryDeviceTimeProjector:
    """Track host-receive minus capture-time as diagnostics; matching uses getTimestamp() directly."""

    def __init__(self, source_name: str, ema_alpha: float = PROJECTOR_EMA_ALPHA):
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
    return raw * 1.6 / (1024.0 ** 3)


def list_available_oaks(status_q: Optional[mp.Queue] = None) -> List[str]:
    devices = dai.Device.getAllAvailableDevices()
    ids = []
    lines = ["Available OAK devices:"]
    for d in devices:
        mxid = d.getDeviceId()
        ids.append(mxid)
        lines.append(f"  {mxid}  state={getattr(d, 'state', 'unknown')}")
    for line in lines:
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

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=sensor_fps)
    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=sensor_fps)
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=sensor_fps)

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


def memory_reader_loop(
    source_name: str,
    q,
    projector: MemoryDeviceTimeProjector,
    store: SharedMemoryStore,
    stop_evt: threading.Event,
    record_evt: mp.Event,
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
            # DepthAI v3 getTimestamp() is already in the host monotonic clock
            # domain. Keep projector stats only as receive-latency diagnostics;
            # do not shift the capture timestamp by USB/pipeline receive delay.
            _ = projector.project(host_ts, dev_ts)
            match_ts = dev_ts if dev_ts is not None else host_ts
            store.append_frame(
                source_name,
                MemFrame(
                    source=source_name,
                    host_ts=host_ts,
                    device_ts=dev_ts,
                    match_ts=match_ts,
                    frame_number=frame_num,
                    rgb=rgb,
                    depth=depth,
                ),
                record=record_evt.is_set(),
            )
        except Exception as e:
            store.add_error(source_name, f"message decode failed: {repr(e)}")
            time.sleep(0.002)


def memory_update_preview(store: SharedMemoryStore, scale: float) -> bool:
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


def append_latest_shared_robot_state(
    shared_robot_state: Dict[str, object],
    robot_states: List[Dict],
    last_robot_ts: Optional[float],
) -> Optional[float]:
    try:
        state = read_shared_robot_state(shared_robot_state)
    except Exception:
        return last_robot_ts
    if not isinstance(state, dict) or "error" in state:
        return last_robot_ts
    ts = float(state.get("host_ts", 0.0))
    if last_robot_ts is None or ts > last_robot_ts + 1e-6:
        robot_states.append(json.loads(json.dumps(state)))
        return ts
    return last_robot_ts


def robot_state_stats(robot_states: List[Dict]) -> Dict:
    meta = {"num_robot_states": len(robot_states)}
    if not robot_states:
        return meta
    ts = np.asarray([float(s["host_ts"]) for s in robot_states], dtype=np.float64)
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
    return meta


def nearest_robot_state(robot_states: List[Dict], target_ts: float) -> Optional[Dict]:
    """Interpolate measured FCI state to the camera capture timestamp.

    Kept under the old function name so the save path stays backward compatible.
    Joint angles and translation are linearly interpolated; RPY uses shortest-angle
    interpolation. Outside the state range, the nearest state is returned and
    marked as extrapolated.
    """
    if not robot_states:
        return None

    ts = np.asarray([float(s["host_ts"]) for s in robot_states], dtype=np.float64)
    j = int(np.searchsorted(ts, float(target_ts), side="left"))

    if j <= 0 or j >= len(robot_states):
        idx = 0 if j <= 0 else len(robot_states) - 1
        state = dict(robot_states[idx])
        err_ms = 1000.0 * (float(state["host_ts"]) - float(target_ts))
        state["sync_target_ts"] = float(target_ts)
        state["sync_error_from_target_ms"] = float(err_ms)
        state["sync_error_abs_ms"] = float(abs(err_ms))
        state["interpolation_extrapolated"] = True
        state["nearest_robot_state_index"] = int(idx)
        return state

    i0, i1 = j - 1, j
    t0, t1 = float(ts[i0]), float(ts[i1])
    if t1 <= t0 + 1e-12:
        alpha = 0.0
    else:
        alpha = float(np.clip((float(target_ts) - t0) / (t1 - t0), 0.0, 1.0))

    s0, s1 = robot_states[i0], robot_states[i1]
    q0 = np.asarray(s0.get("joint_angles_rad", [0.0] * 7), dtype=float)
    q1 = np.asarray(s1.get("joint_angles_rad", [0.0] * 7), dtype=float)
    q = (1.0 - alpha) * q0 + alpha * q1

    p0 = np.asarray(s0.get("pose_xyzrpy", [0.0] * 6), dtype=float)
    p1 = np.asarray(s1.get("pose_xyzrpy", [0.0] * 6), dtype=float)
    pose = np.empty(6, dtype=float)
    pose[:3] = (1.0 - alpha) * p0[:3] + alpha * p1[:3]
    dang = _wrap_angle_rad(p1[3:6] - p0[3:6])
    pose[3:6] = _wrap_angle_rad(p0[3:6] + alpha * dang)

    nearest_err_ms = 1000.0 * min(abs(float(target_ts) - t0), abs(t1 - float(target_ts)))
    return {
        "host_ts": float(target_ts),
        "pose_xyzrpy": pose.tolist(),
        "joint_angles_rad": q.tolist(),
        "state_source": "fci_state_interpolated_to_camera_capture",
        "sync_target_ts": float(target_ts),
        "sync_error_from_target_ms": 0.0,
        "sync_error_abs_ms": float(nearest_err_ms),
        "interpolation_extrapolated": False,
        "interpolation_alpha": float(alpha),
        "interpolation_left_index": int(i0),
        "interpolation_right_index": int(i1),
        "interpolation_left_ts": t0,
        "interpolation_right_ts": t1,
        "interpolation_bracket_gap_ms": float(1000.0 * (t1 - t0)),
    }


def _frame_arrays(frames_by_source: Dict[str, List[MemFrame]]) -> Dict[str, Dict[str, np.ndarray]]:
    return {name: {"ts": np.asarray([f.match_ts for f in frames], dtype=np.float64)} for name, frames in frames_by_source.items()}


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
            cand = [k for k in (j - 1, j, j + 1) if 0 <= k < ts.size]
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
                fail_reason = f"{name} nearest error {err_s * 1000.0:.2f}ms > tolerance"
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


def memory_save_bundle(
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


def memory_save_all_bundles(
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
    status_put(status_q, f"Saving {len(groups)} synchronized bundles to {out_root} ...") if status_q is not None else print(
        f"Saving {len(groups)} synchronized bundles to {out_root} ..."
    )
    t0 = time.monotonic()
    robot_errors = []
    for i, (group, target_ts) in enumerate(zip(groups, target_times)):
        robot = nearest_robot_state(robot_states, target_ts)
        if robot is not None and "sync_error_abs_ms" in robot:
            robot_errors.append(float(robot["sync_error_abs_ms"]))
        picked_ts = [fr.match_ts for fr in group.values()]
        extra_meta = {
            **args_meta,
            "projectors": projectors_meta,
            "per_source_match_offsets_ms": {
                name: 1000.0 * (fr.match_ts - min(picked_ts)) for name, fr in group.items()
            },
            "offline_sync_meta_preview": {
                "sync_fps": sync_meta.get("sync_fps"),
                "sync_tolerance_ms": sync_meta.get("sync_tolerance_ms"),
                "period_s": sync_meta.get("period_s"),
                "chosen_phase_s": sync_meta.get("chosen_phase_s"),
            },
        }
        memory_save_bundle(i, out_root, group, target_ts, robot, extra_meta)
        if i % 20 == 0:
            status_put(status_q, f"  saved bundle {i:06d}/{len(groups)}") if status_q is not None else print(
                f"  saved bundle {i:06d}/{len(groups)}"
            )
    dt = time.monotonic() - t0
    summary = {"save_duration_s": dt, "num_saved_bundles": len(groups)}
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


def recorder_process_main(
    args: argparse.Namespace,
    record_evt: mp.Event,
    stop_evt: mp.Event,
    status_q: mp.Queue,
    shared_robot_state: Dict[str, object],
    camera_ready_evt: Optional[mp.Event] = None,
) -> None:
    """Memory-first 3-OAK recorder used by the AnyGrasp lift executor.

    During recording, every OAK RGB-D frame is kept in CPU RAM and robot states
    are sampled from shared memory.  After record_evt is cleared, the process
    performs offline fixed-rate synchronization and writes the original bundle
    directory structure.
    """
    global FPS
    FPS = int(args.record_fps)
    apply_record_resolution(args.record_resolution)
    rgb_size = (OAK_W, OAK_H)
    mono_size = (OAK_W, OAK_H)
    oak_ids = list(args.record_oak_ids)
    source_names = [f"oak_{mxid}" for mxid in oak_ids]
    reader_stop_evt = threading.Event()
    robot_states: List[Dict] = []
    last_robot_ts: Optional[float] = None
    out_root: Optional[Path] = None
    start_wall: Optional[str] = None

    try:
        max_duration = float(args.record_max_duration)
        if max_duration <= 0:
            raise ValueError("--record-max-duration must be positive")
        if max_duration > HARD_MEMORY_MAX_DURATION_S:
            raise ValueError(
                f"For safety, --record-max-duration is capped at {HARD_MEMORY_MAX_DURATION_S:.1f}s. "
                f"You requested {max_duration:.1f}s."
            )

        with contextlib.ExitStack() as exit_stack:
            oaks = open_oaks_v3(
                exit_stack=exit_stack,
                oak_ids=oak_ids,
                output_fps=int(args.record_fps),
                sensor_fps=int(args.record_sensor_fps),
                rgb_size=rgb_size,
                mono_size=mono_size,
                oak_internal_sync_ms=float(args.oak_internal_sync_ms),
                enable_undistortion=bool(args.enable_undistortion),
                status_q=status_q,
            )
            projectors = {
                info["source_name"]: MemoryDeviceTimeProjector(info["source_name"])
                for info in oaks.values()
            }
            store = SharedMemoryStore(source_names)
            threads = []
            for _mxid, info in oaks.items():
                name = info["source_name"]
                t = threading.Thread(
                    target=memory_reader_loop,
                    args=(name, info["queue"], projectors[name], store, reader_stop_evt, record_evt),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            warm_s = max(0.0, float(args.camera_warmup_seconds))
            status_put(status_q, f"Memory recorder warming OAK streams for {warm_s:.2f}s...")
            warmup_end = time.monotonic() + warm_s
            while time.monotonic() < warmup_end and not stop_evt.is_set():
                last_robot_ts = append_latest_shared_robot_state(shared_robot_state, robot_states, last_robot_ts)
                if args.record_preview:
                    if not memory_update_preview(store, float(args.record_preview_scale)):
                        stop_evt.set()
                        break
                time.sleep(0.01)

            status_put(status_q, "Freezing OAK receive-latency diagnostics (capture timestamps are used directly):")
            for name, prj in projectors.items():
                prj.freeze(float(args.record_freeze_percentile))
                status_put(status_q, f"  {name}: diagnostic_offset={prj.fixed_offset:.6f}, samples={len(prj.offset_samples)}")

            avail_gib = get_mem_available_gib()
            est_gib = estimate_ram_gib(len(oak_ids), rgb_size, float(args.record_fps), max_duration)
            status_put(status_q, "========== 3-OAK memory-first recorder ready ==========")
            status_put(status_q, f"Resolution: {args.record_resolution} RGB {rgb_size} mono/depth {mono_size}")
            status_put(status_q, f"FPS: output={args.record_fps}, sensor={args.record_sensor_fps}, offline sync={args.record_sync_fps}")
            status_put(status_q, f"Max duration: {max_duration:.2f}s")
            status_put(status_q, f"Estimated RAM for raw OAK frames: ~{est_gib:.1f} GiB")
            if avail_gib is not None:
                status_put(status_q, f"Current MemAvailable: {avail_gib:.1f} GiB")
                if est_gib > 0.75 * avail_gib:
                    status_put(status_q, "WARNING: estimated memory is high compared with available RAM.")

            if camera_ready_evt is not None:
                camera_ready_evt.set()
            status_put(status_q, "Recorder cameras ready after memory warmup. Waiting for robot lift record_evt.")

            while not stop_evt.is_set() and not record_evt.is_set():
                last_robot_ts = append_latest_shared_robot_state(shared_robot_state, robot_states, last_robot_ts)
                if args.record_preview:
                    if not memory_update_preview(store, float(args.record_preview_scale)):
                        stop_evt.set()
                        break
                time.sleep(0.01)

            if stop_evt.is_set() and not record_evt.is_set():
                status_put(status_q, "Recorder stopped before recording started.")
                return

            store.clear_recorded()
            robot_states.clear()
            last_robot_ts = None
            out_root = Path(args.record_output_root) if args.record_output_root is not None else Path.cwd() / f"{SAVE_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if out_root.exists() and not args.overwrite_recording:
                existing = list(out_root.glob("bundle_*"))
                if existing or (out_root / "recording_summary.json").exists() or (out_root / "session_summary.json").exists():
                    raise RuntimeError(
                        f"Output folder {out_root} already contains recording files. Use --overwrite-recording or choose a new folder."
                    )
            ensure_dir(out_root)
            start_wall = datetime.now().isoformat()
            record_start = time.monotonic()
            record_end = record_start + max_duration
            last_report = record_start
            status_put(status_q, f"Start MEMORY recording -> {out_root}")
            status_put(status_q, "No online sync/save decisions will be made during recording.")

            while not stop_evt.is_set() and record_evt.is_set():
                last_robot_ts = append_latest_shared_robot_state(shared_robot_state, robot_states, last_robot_ts)
                now = time.monotonic()
                if now - last_report >= 1.0:
                    counts = store.get_record_counts()
                    status_put(status_q, f"[memory-record] t={now - record_start:.1f}s counts={counts} robot_states={len(robot_states)}")
                    last_report = now
                if args.record_preview:
                    if not memory_update_preview(store, float(args.record_preview_scale)):
                        record_evt.clear()
                        stop_evt.set()
                        break
                if now >= record_end:
                    status_put(status_q, f"[memory-record] reached --record-max-duration={max_duration:.1f}s; stopping recording.")
                    record_evt.clear()
                    break
                time.sleep(0.005)

            # Capture a few final robot states after the event clears, if still available.
            post_t0 = time.monotonic()
            while time.monotonic() - post_t0 < 0.25:
                last_robot_ts = append_latest_shared_robot_state(shared_robot_state, robot_states, last_robot_ts)
                time.sleep(0.01)

            frames_by_source = store.get_frames_copy()
            for name in source_names:
                status_put(status_q, f"Recorded {name}: {len(frames_by_source[name])} frames")
            if any(len(frames_by_source[name]) == 0 for name in source_names):
                raise RuntimeError(f"At least one OAK recorded zero frames: { {n: len(frames_by_source[n]) for n in source_names} }")

            groups, target_times, sync_meta = offline_fixed_rate_sync(
                frames_by_source=frames_by_source,
                source_names=source_names,
                sync_fps=float(args.record_sync_fps),
                tolerance_ms=float(args.record_sync_tolerance_ms),
                phase_steps=int(args.record_phase_steps),
                allow_reuse=bool(args.record_allow_reuse),
            )
            if not groups:
                raise RuntimeError("Offline synchronization produced zero bundles. Try larger --record-sync-tolerance-ms.")

            projectors_meta = {name: prj.as_meta() for name, prj in projectors.items()}
            args_meta = {
                "record_resolution": args.record_resolution,
                "record_fps": int(args.record_fps),
                "record_sensor_fps": int(args.record_sensor_fps),
                "record_sync_fps": float(args.record_sync_fps),
                "record_sync_tolerance_ms": float(args.record_sync_tolerance_ms),
                "record_oak_ids": list(oak_ids),
                "memory_first_recording": True,
                "offline_sync_mode": "fixed-rate",
                "robot_state_source": "robot_process_single_fr3_connection_shared_memory_actual_pose_and_joints",
                "robot_calibration_copy": getattr(args, "robot_calibration_copy", None),
                "storage_layout": "bundle_XXXXXX/oak_<MXID>/{rgb.jpg,depth.npy} + meta.json",
            }
            save_summary = memory_save_all_bundles(
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
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "output_root": str(out_root),
                "raw_frame_counts": {name: len(frames_by_source[name]) for name in source_names},
                "record_counts": store.record_counts,
                "raw_counts_including_warmup": store.raw_counts,
                "reader_errors": store.errors,
                "projectors": projectors_meta,
                "offline_sync": sync_meta,
                "robot_states": robot_state_stats(robot_states),
                "save_summary": save_summary,
                "record_resolution": args.record_resolution,
                "fps": int(args.record_fps),
                "oak_resolution": [OAK_W, OAK_H],
                "calibration_manifest": build_calibration_manifest(),
                "robot_calibration_copy": getattr(args, "robot_calibration_copy", None),
                "stopped_at_wallclock": datetime.now().isoformat(),
            }
            with open(out_root / "recording_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            # Keep the previous execute_* recorder summary filename too.
            with open(out_root / "session_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            status_put(status_q, "Done saving memory-first recording.")
            status_put(status_q, f"Saved {len(groups)} synchronized bundles to {out_root.resolve()}")
            status_put(status_q, f"Summary: {out_root / 'recording_summary.json'}")

    except KeyboardInterrupt:
        stop_evt.set()
        raise
    except Exception as e:
        status_put(status_q, f"[recorder] ERROR: {type(e).__name__}: {e}")
        stop_evt.set()
        raise
    finally:
        try:
            reader_stop_evt.set()
            if args.record_preview:
                cv2.destroyAllWindows()
        except Exception:
            pass
        stop_evt.set()
        status_put(status_q, "Recorder process exiting.")



# =========================
# AnyGrasp robot process: direct-FCI version
# =========================
def print_all_joints_from_state(status_q: mp.Queue, state: Optional[Dict], stage: str) -> None:
    if not state or "joint_angles_rad" not in state:
        status_put(status_q, f"[safety] {stage}: joint state unavailable")
        return
    q = np.asarray(state["joint_angles_rad"], dtype=np.float64).reshape(-1)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        status_put(status_q, f"[safety] {stage}: invalid joint state {q}")
        return
    joints = " ".join(f"J{i + 1}={np.degrees(value):+.1f}deg" for i, value in enumerate(q))
    status_put(status_q, f"[safety] {stage}: {joints}")


def _wrap_angle_rad(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def pose_error_xyzrpy(current_pose: List[float], target_pose: List[float]) -> Tuple[float, float]:
    cur = np.asarray(current_pose, dtype=float)
    tgt = np.asarray(target_pose, dtype=float)
    pos_err = float(np.linalg.norm(cur[:3] - tgt[:3]))
    rot_err = float(np.linalg.norm(_wrap_angle_rad(cur[3:6] - tgt[3:6])))
    return pos_err, rot_err


def robot_checkpoint(
    status_q: mp.Queue,
    message: str,
    auto_yes: bool,
    console_lock=None,
) -> None:
    """Interactive checkpoint: press Enter to continue, or use -y/--yes.

    The robot sequence runs in a child process, so read from /dev/tty rather than
    inherited stdin. For backward compatibility, typing "yes" or "y" also
    continues; any other non-empty response aborts.
    """
    if auto_yes:
        status_put(status_q, f"[checkpoint][auto-yes] {message}")
        return

    if console_lock is not None:
        console_lock.acquire()
    try:
        fd_in = os.open("/dev/tty", os.O_RDONLY)
        fd_out = os.open("/dev/tty", os.O_WRONLY)
        with os.fdopen(fd_out, "w", encoding="utf-8", buffering=1) as tty_out:
            tty_out.write(
                "\n"
                "============================================================\n"
                "ROBOT CHECKPOINT\n"
                f"{message}\n"
                "Press Enter to continue (y/yes also works); anything else aborts.\n"
                "> "
            )
            tty_out.flush()
        with os.fdopen(fd_in, "r", encoding="utf-8", buffering=1) as tty_in:
            ans = tty_in.readline().strip().lower()
    except Exception as e:
        raise RuntimeError(
            f"Cannot read checkpoint input from /dev/tty ({type(e).__name__}: {e}). "
            f"Run from an interactive terminal, or pass -y/--yes."
        ) from e
    finally:
        if "ans" in locals() and ans in {"", "yes", "y"}:
            try:
                fd_out = os.open("/dev/tty", os.O_WRONLY)
                with os.fdopen(fd_out, "w", encoding="utf-8", buffering=1) as tty_out:
                    tty_out.write(f"[confirmed] {message}\n")
                    tty_out.write("============================================================\n\n")
                    tty_out.flush()
            except Exception:
                # Confirmation output is cosmetic; never fail a confirmed
                # checkpoint because the terminal disappeared after input.
                pass
        if console_lock is not None:
            console_lock.release()

    if ans not in {"", "yes", "y"}:
        raise RuntimeError(f"User aborted at checkpoint: {message}")


def _fci_state_to_dict(state, host_ts: Optional[float] = None, *, stage: str = "") -> Dict:
    T = np.asarray(state.O_T_EE, dtype=float).reshape(4, 4, order="F")
    rpy = grasp_exec.rpy_from_matrix(T[:3, :3])
    out = {
        "host_ts": float(time.monotonic() if host_ts is None else host_ts),
        "pose_xyzrpy": [
            float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
            float(rpy[0]), float(rpy[1]), float(rpy[2]),
        ],
        "joint_angles_rad": np.asarray(state.q, dtype=float).tolist(),
        "control_command_success_rate": float(getattr(state, "control_command_success_rate", 1.0)),
        "state_source": "direct_fci_robot_state",
    }
    if stage:
        out["stage"] = stage
    return out


def _publish_raw_robot_state(robot, shared_robot_state: Dict[str, object], *, stage: str = "") -> Optional[Dict]:
    try:
        state = robot.read_once()
        out = _fci_state_to_dict(state, stage=stage)
        write_shared_robot_state(shared_robot_state, out)
        return out
    except Exception:
        return None


FR3_SOFT_LIMITS_RAD = np.asarray(
    [
        [-2.8007, 2.8007],
        [-1.7361, 1.7361],
        [-2.8007, 2.8007],
        [-2.7500, -0.2200],
        [-2.7763, 2.7763],
        [0.5398, 4.5216],
        [-2.9508, 2.9508],
    ],
    dtype=np.float64,
)


def _validate_pyroki_hover_target(
    grasp_record: Dict,
    expected_target_T: np.ndarray,
    *,
    max_position_error_m: float = 0.001,
    max_orientation_error_deg: float = 0.5,
) -> None:
    """Reject stale safe JSON generated with different hover/calibration options."""
    try:
        stored_values = grasp_record["table_ik_fk_check"]["ik_fk_endpoints"]["hover"][
            "target_T_base_from_tcp"
        ]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Selected grasp lacks the PyRoki hover target transform. "
            "Regenerate grasps_table_ik_safe.json with the current filter."
        ) from exc
    stored = np.asarray(stored_values, dtype=np.float64)
    expected = np.asarray(expected_target_T, dtype=np.float64)
    if stored.shape != (4, 4) or expected.shape != (4, 4):
        raise RuntimeError("PyRoki hover target transform must be 4x4")
    position_error = float(np.linalg.norm(stored[:3, 3] - expected[:3, 3]))
    relative_R = stored[:3, :3] @ expected[:3, :3].T
    rvec, _ = cv2.Rodrigues(relative_R)
    orientation_error_deg = float(np.degrees(np.linalg.norm(rvec)))
    if position_error > float(max_position_error_m) or orientation_error_deg > float(max_orientation_error_deg):
        raise RuntimeError(
            "Stored PyRoki hover validation was generated for a different target/configuration: "
            f"position_error={position_error:.6f}m, "
            f"orientation_error={orientation_error_deg:.4f}deg. "
            "Regenerate the PyRoki safe grasp file before Cartesian execution."
        )


def _minimum_joint_soft_margin(q: np.ndarray) -> Tuple[float, int]:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise RuntimeError(f"Expected 7 finite joint positions, got {q}")
    margins = np.minimum(q - FR3_SOFT_LIMITS_RAD[:, 0], FR3_SOFT_LIMITS_RAD[:, 1] - q)
    worst = int(np.argmin(margins))
    return float(margins[worst]), worst


def _check_joint_soft_limits(q: np.ndarray, margin_rad: float, *, label: str) -> None:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    _minimum_joint_soft_margin(q)
    margin = float(margin_rad)
    if margin < 0.0:
        raise RuntimeError("Joint-limit margin must be non-negative")
    lower = FR3_SOFT_LIMITS_RAD[:, 0] + margin
    upper = FR3_SOFT_LIMITS_RAD[:, 1] - margin
    bad = np.flatnonzero((q < lower) | (q > upper))
    if bad.size:
        details = ", ".join(
            f"J{i + 1}={q[i]:+.4f} not in [{lower[i]:+.4f}, {upper[i]:+.4f}]"
            for i in bad
        )
        raise RuntimeError(f"{label}: joint soft-limit margin check failed: {details}")


def _direct_cartesian_move_fci(
    robot,
    shared_robot_state: Dict[str, object],
    status_q: mp.Queue,
    *,
    name: str,
    target_pose: List[float],
    speed: float,
    state_publish_hz: float,
    max_angular_speed: float,
    joint_limit_margin_rad: float,
    stop_evt: Optional[mp.Event] = None,
    settle_s: float = 0.0,
    require_motion_start: bool = False,
    motion_start_delta_m: float = 0.0015,
    motion_start_timeout_s: float = 3.0,
    on_motion_started=None,
) -> Optional[Dict]:
    """Move Cartesian pose directly in the process's 1 kHz FCI loop.

    This avoids fr3.ControlThread + same-process polling. The same readOnce()
    that drives control also supplies the measured state published for recording.
    """
    from pylibfranka import CartesianPose, ControllerMode

    x, y, z, roll, pitch, yaw = [float(v) for v in target_pose]
    target_T = np.eye(4, dtype=np.float64)
    target_T[:3, :3] = grasp_exec.rpy_to_matrix(roll, pitch, yaw)
    target_T[:3, 3] = [x, y, z]

    status_put(
        status_q,
        f"[motion] {name} direct-FCI -> ({x:+.4f}, {y:+.4f}, {z:+.4f}) speed={float(speed):.4f}",
    )

    active = robot.start_cartesian_pose_control(ControllerMode.CartesianImpedance)
    last_state = None
    min_success = 1.0
    max_compute_us = 0.0
    next_publish_ts: Optional[float] = None
    publish_period = 1.0 / max(float(state_publish_hz), 1.0)
    motion_started = False
    start_wall = time.monotonic()

    try:
        state, _ = active.readOnce()
        state_ts = time.monotonic()
        last_state = state
        start_T = np.asarray(state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F").copy()
        start_xyz = start_T[:3, 3].copy()
        _check_joint_soft_limits(
            np.asarray(state.q, dtype=np.float64),
            joint_limit_margin_rad,
            label=f"{name} Cartesian start",
        )
        write_shared_robot_state(shared_robot_state, _fci_state_to_dict(state, state_ts, stage=f"before_{name}"))
        next_publish_ts = state_ts

        dp = target_T[:3, 3] - start_T[:3, 3]
        dist = float(np.linalg.norm(dp))
        R_rel = target_T[:3, :3] @ start_T[:3, :3].T
        rvec, _ = cv2.Rodrigues(R_rel)
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        rot_angle = float(np.linalg.norm(rvec))

        linear_t = dist / max(float(speed), 1e-4)
        angular_t = rot_angle / max(float(max_angular_speed), 1e-3)
        total_t = max(0.35, linear_t, angular_t)
        status_put(status_q, f"[motion] {name} planned duration={total_t:.3f}s dist={dist:.4f}m rot={rot_angle:.4f}rad")

        elapsed = 0.0
        last_report = time.monotonic()
        while elapsed < total_t:
            if stop_evt is not None and stop_evt.is_set():
                raise RuntimeError(f"stop event set during {name}")

            state, duration = active.readOnce()
            state_host_ts = time.monotonic()
            compute_t0 = time.perf_counter_ns()
            last_state = state

            try:
                dt = float(duration.to_sec())
            except Exception:
                dt = 0.001
            if not np.isfinite(dt) or dt <= 0.0 or dt > 0.02:
                dt = 0.001
            elapsed += dt

            if next_publish_ts is None or state_host_ts + 1e-12 >= next_publish_ts:
                write_shared_robot_state(
                    shared_robot_state,
                    _fci_state_to_dict(state, state_host_ts, stage=f"{name}_fci"),
                )
                while next_publish_ts is not None and next_publish_ts <= state_host_ts:
                    next_publish_ts += publish_period

            success = float(getattr(state, "control_command_success_rate", 1.0))
            min_success = min(min_success, success)

            actual_T = np.asarray(state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F")
            _check_joint_soft_limits(
                np.asarray(state.q, dtype=np.float64),
                joint_limit_margin_rad,
                label=f"{name} Cartesian runtime",
            )
            moved = float(np.linalg.norm(actual_T[:3, 3] - start_xyz))
            if (not motion_started) and moved >= float(motion_start_delta_m):
                motion_started = True
                status_put(status_q, f"[motion] {name} measured motion started: moved={moved:.4f} m")
                if on_motion_started is not None:
                    on_motion_started()
            if require_motion_start and (not motion_started) and (time.monotonic() - start_wall > float(motion_start_timeout_s)):
                raise RuntimeError(
                    f"{name}: measured TCP did not move >{float(motion_start_delta_m):.4f}m "
                    f"within {float(motion_start_timeout_s):.1f}s"
                )

            u = min(1.0, max(0.0, elapsed / total_t))
            s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            cmd_T = np.eye(4, dtype=np.float64)
            cmd_T[:3, 3] = start_T[:3, 3] + s * dp
            if rot_angle > 1e-10:
                dR, _ = cv2.Rodrigues(rvec * s)
                cmd_T[:3, :3] = dR @ start_T[:3, :3]
            else:
                cmd_T[:3, :3] = start_T[:3, :3]

            cmd = CartesianPose(list(cmd_T.flatten(order="F")))
            active.writeOnce(cmd)

            compute_us = (time.perf_counter_ns() - compute_t0) / 1000.0
            max_compute_us = max(max_compute_us, compute_us)
            if time.monotonic() - last_report >= 1.0:
                minimum_margin, worst_joint = _minimum_joint_soft_margin(state.q)
                status_put(
                    status_q,
                    f"[motion] {name}: {100.0*u:.0f}% success_rate={success:.3f} "
                    f"worst_margin=J{worst_joint + 1}:{minimum_margin:.3f}rad",
                )
                last_report = time.monotonic()

        # Finish exactly at target with a motion_finished command.
        state, _ = active.readOnce()
        state_host_ts = time.monotonic()
        last_state = state
        write_shared_robot_state(shared_robot_state, _fci_state_to_dict(state, state_host_ts, stage=f"after_{name}"))
        final_cmd = CartesianPose(list(target_T.flatten(order="F")))
        final_cmd.motion_finished = True
        active.writeOnce(final_cmd)

    finally:
        try:
            robot.stop()
        except Exception:
            pass

    if float(settle_s) > 0.0:
        settle_end = time.monotonic() + float(settle_s)
        while time.monotonic() < settle_end:
            if stop_evt is not None and stop_evt.is_set():
                break
            _publish_raw_robot_state(robot, shared_robot_state, stage=f"{name}_settle")
            time.sleep(min(0.01, publish_period))

    final_state = _publish_raw_robot_state(robot, shared_robot_state, stage=f"after_{name}_final")
    if final_state is None and last_state is not None:
        final_state = _fci_state_to_dict(last_state, stage=f"after_{name}_last_fci")
        write_shared_robot_state(shared_robot_state, final_state)

    if final_state is not None:
        pos_err, rot_err = pose_error_xyzrpy(final_state["pose_xyzrpy"], target_pose)
        status_put(
            status_q,
            f"[motion] {name} finished: pos_err={pos_err:.4f}m rot_err={rot_err:.4f}rad "
            f"min_success={min_success:.3f} max_compute={max_compute_us:.1f}us",
        )
    return final_state


def anygrasp_robot_process_main(
    record_evt: mp.Event,
    stop_evt: mp.Event,
    status_q: mp.Queue,
    shared_robot_state: Dict[str, object],
    args_dict: Dict,
    target_pose: List[float],
    grasp_record: Dict,
    camera_ready_evt: Optional[mp.Event] = None,
    console_lock=None,
) -> None:
    """Execute hover/descend/lift with direct Cartesian FCI control."""
    # Keep helper BLAS pools out of the realtime process.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    from pylibfranka import Robot
    from fr3._gripper import GripperControl

    ip = args_dict.get("ip", ROBOT_IP)
    fr3.ROBOT_IP = ip

    x, y, z, roll, pitch, yaw = [float(v) for v in target_pose]
    hover_pose = [x, y, z + float(args_dict["hover_z_offset"]), roll, pitch, yaw]
    final_pose = [x, y, z + float(args_dict["final_z_offset"]), roll, pitch, yaw]
    lift_pose = [x, y, z + float(args_dict["lift_z_offset"]), roll, pitch, yaw]

    robot = None
    gripper = None
    try:
        if camera_ready_evt is not None:
            status_put(status_q, "[robot] waiting for recorder cameras ready...")
            if not camera_ready_evt.wait(timeout=90.0):
                raise RuntimeError("recorder cameras were not ready within 90s")
            status_put(status_q, "[robot] recorder cameras ready")

        robot_checkpoint(
            status_q, "Connect to robot and start?",
            bool(args_dict.get("yes", False)), console_lock,
        )

        # Use the existing high-level home routine only before opening the
        # persistent low-level arm/gripper connections.
        if args_dict.get("home", False):
            robot_checkpoint(
                status_q, "Home the robot?",
                bool(args_dict.get("yes", False)), console_lock,
            )
            status_put(status_q, "[robot] homing...")
            with fr3.FR3(ip=ip) as home_wrapper:
                home_wrapper.home(blocking=True)

        robot = Robot(ip)
        robot.automatic_error_recovery()
        robot.set_collision_behavior(
            [20.0] * 7, [40.0] * 7,
            [10.0] * 6, [20.0] * 6,
        )
        robot.set_joint_impedance([3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0])
        robot.set_cartesian_impedance([3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0])
        gripper = GripperControl(ip)
        status_put(status_q, "[robot] Cartesian direct-FCI arm + separate gripper connected")

        initial = _publish_raw_robot_state(robot, shared_robot_state, stage="after_connect")
        print_all_joints_from_state(status_q, initial, "after_connect")

        if args_dict.get("skip_open_gripper", False):
            status_put(status_q, "[robot] skipping gripper open before hover")
        else:
            status_put(status_q, "[robot] opening gripper...")
            gripper.open(speed=0.1)

        prelift_settle_s = float(args_dict.get("prelift_settle_s", 0.30))
        state_hz = float(args_dict.get("state_publish_hz", 100.0))
        max_ang = float(args_dict.get("servo_max_angular", 0.4))

        robot_checkpoint(
            status_q, "Move to HOVER pose with Cartesian control?",
            bool(args_dict.get("yes", False)), console_lock,
        )
        latest_state = _direct_cartesian_move_fci(
            robot, shared_robot_state, status_q,
            name="hover", target_pose=hover_pose,
            speed=float(args_dict.get("motion_speed", 0.10)),
            state_publish_hz=state_hz, max_angular_speed=max_ang,
            joint_limit_margin_rad=float(args_dict.get("cartesian_runtime_joint_margin_rad", 0.15)),
            stop_evt=stop_evt, settle_s=prelift_settle_s,
        )
        print_all_joints_from_state(status_q, latest_state, "after_hover")

        if args_dict.get("hover_only", False):
            status_put(status_q, "[done] hover-only mode; stopping")
            return

        if not args_dict.get("skip_open_gripper", False):
            gripper.open(speed=0.1)
            time.sleep(0.10)

        robot_checkpoint(
            status_q, "Descend to FINAL grasp pose?",
            bool(args_dict.get("yes", False)), console_lock,
        )
        latest_state = _direct_cartesian_move_fci(
            robot, shared_robot_state, status_q,
            name="descend", target_pose=final_pose,
            speed=float(args_dict.get("descend_speed", 0.03)),
            state_publish_hz=state_hz, max_angular_speed=max_ang,
            joint_limit_margin_rad=float(args_dict.get("cartesian_runtime_joint_margin_rad", 0.15)),
            stop_evt=stop_evt, settle_s=prelift_settle_s,
        )
        print_all_joints_from_state(status_q, latest_state, "after_descend")

        grasp_width = float(args_dict.get("grasp_width", 0.0))
        if args_dict.get("use_anygrasp_width", False):
            try:
                w = grasp_exec.get_record_width(grasp_record)
            except Exception:
                w = None
            if w is not None:
                grasp_width = max(0.0, min(float(w), float(args_dict.get("max_gripper_width", 0.08))))
            else:
                status_put(status_q, "[grasp][WARN] AnyGrasp width unavailable; using --grasp-width")

        robot_checkpoint(
            status_q, "Close gripper and grasp?",
            bool(args_dict.get("yes", False)), console_lock,
        )
        status_put(status_q, f"[grasp] width={grasp_width:.4f}m force={float(args_dict.get('grasp_force', 35.0)):.1f}N")
        got = gripper.grasp(
            width=grasp_width,
            speed=float(args_dict.get("grasp_close_speed", 0.04)),
            force=float(args_dict.get("grasp_force", 35.0)),
            epsilon_inner=float(args_dict.get("grasp_epsilon", 0.08)),
            epsilon_outer=float(args_dict.get("grasp_epsilon", 0.08)),
        )
        gs = gripper.read_state()
        status_put(status_q, f"[grasp] reported={'SUCCESS' if got else 'FAILED'} actual_width={gs['width']*1000.0:.1f}mm is_grasped={gs['is_grasped']}")
        _publish_raw_robot_state(robot, shared_robot_state, stage="after_grasp")

        if (not bool(args_dict.get("lift_even_if_not_grasped", False))) and (not bool(gs.get("is_grasped", False))):
            status_put(status_q, "[abort] gripper reports no object; skipping lift")
            gripper.open(speed=0.1)
            return

        def _start_recording_after_lift_motion_started():
            if not record_evt.is_set():
                status_put(status_q, "[record] toggling recorder ON after measured lift motion started")
                record_evt.set()

        robot_checkpoint(
            status_q, "Lift object?",
            bool(args_dict.get("yes", False)), console_lock,
        )
        latest_state = _direct_cartesian_move_fci(
            robot, shared_robot_state, status_q,
            name="lift", target_pose=lift_pose,
            speed=float(args_dict.get("lift_speed", 0.05)),
            state_publish_hz=state_hz, max_angular_speed=max_ang,
            joint_limit_margin_rad=float(args_dict.get("cartesian_runtime_joint_margin_rad", 0.15)),
            stop_evt=stop_evt, settle_s=0.0,
            require_motion_start=True,
            motion_start_delta_m=float(args_dict.get("motion_start_delta_m", 0.0015)),
            motion_start_timeout_s=float(args_dict.get("motion_start_timeout_s", 3.0)),
            on_motion_started=_start_recording_after_lift_motion_started,
        )
        print_all_joints_from_state(status_q, latest_state, "after_lift")

        after_lift = float(args_dict.get("record_after_lift", 1.0))
        if after_lift > 0.0 and record_evt.is_set():
            status_put(status_q, f"[record] keep recording after lift: {after_lift:.2f}s")
            end_t = time.monotonic() + after_lift
            period = 1.0 / max(state_hz, 1.0)
            while time.monotonic() < end_t and not stop_evt.is_set():
                _publish_raw_robot_state(robot, shared_robot_state, stage="post_lift_recording")
                time.sleep(period)

        record_evt.clear()
        status_put(status_q, "[record] toggling recorder OFF")

        try:
            gs2 = gripper.read_state()
            status_put(status_q, f"[post-lift] gripper width={gs2['width']*1000.0:.1f}mm is_grasped={gs2['is_grasped']}")
        except Exception as e:
            status_put(status_q, f"[post-lift][WARN] gripper state failed: {type(e).__name__}: {e}")

    except KeyboardInterrupt:
        stop_evt.set()
        raise
    except Exception as e:
        status_put(status_q, f"[robot] ERROR: {type(e).__name__}: {e}")
        stop_evt.set()
        raise
    finally:
        record_evt.clear()
        if gripper is not None:
            try:
                gripper.stop()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        status_put(status_q, "Robot process exiting.")


# =========================
# Argument parsing
# =========================
def add_recording_and_servo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--record-resolution",
        choices=("low", "high"),
        default="high",
        help="RGB-D recording resolution. low = OAK 640x400; high = OAK 1280x720.",
    )
    # Alias for execute_anygrasp_franka.py --yes, requested for one-command full execution.
    parser.add_argument("-y", dest="yes", action="store_true", help="Alias for --yes; auto-confirm all checkpoints. Without it, press Enter at each checkpoint.")
    parser.add_argument(
        "--record-fps", type=int, default=FPS,
        help="Recording output FPS for each OAK stream. Keep 15 if 10 FPS gives black frames."
    )
    parser.add_argument("--record-sensor-fps", type=int, default=FPS,
                        help="OAK sensor FPS. Usually keep 15.")
    parser.add_argument("--record-sync-fps", type=float, default=float(FPS),
                        help="Offline fixed-rate synchronized output FPS.")
    parser.add_argument("--record-sync-tolerance-ms", type=float, default=float(SYNC_TOLERANCE_MS),
                        help="Offline synchronization tolerance in milliseconds.")
    parser.add_argument("--record-oak-ids", nargs="+", default=list(TARGET_OAK_IDS),
                        help="OAK MXIDs to record; defaults to $AURA_OAK_IDS, camera-index order.")
    parser.add_argument("--oak-internal-sync-ms", type=float, default=float(OAK_INTERNAL_SYNC_MS),
                        help="DepthAI internal RGB-depth sync threshold in milliseconds.")
    parser.add_argument("--enable-undistortion", type=str2bool, default=True, metavar="True|False",
                        help="Enable DepthAI RGB undistortion in the camera pipeline.")
    parser.add_argument("--record-freeze-percentile", type=float, default=5.0,
                        help="Percentile used only for receive-latency diagnostics; camera matching uses DepthAI getTimestamp() directly.")
    parser.add_argument("--record-phase-steps", type=int, default=120,
                        help="Number of candidate phases tested within one offline-sync period.")
    parser.add_argument("--record-allow-reuse", type=str2bool, default=False, metavar="True|False",
                        help="Allow reusing a camera frame for adjacent output timesteps. Usually False for dynamics.")
    parser.add_argument("--record-preview", type=str2bool, default=False, metavar="True|False",
                        help="Show OAK RGB preview in the recorder child.")
    parser.add_argument("--record-preview-scale", type=float, default=PREVIEW_SCALE)
    parser.add_argument("--record-output-root", type=Path, default=None,
                        help="Output folder for lift recording; pass it explicitly (the one-shot uses <case>/phystwin/raw_recording). Default: <dir>/phystwin for the first ancestor <dir> of --grasp-json whose parent is named 'data', which assumes a flat data/<case>/ layout.")
    parser.add_argument("--overwrite-recording", action="store_true",
                        help="Allow using an existing --record-output-root.")
    parser.add_argument("--record-max-duration", type=float, default=60.0,
                        help="Safety cap for memory-first recording if record_evt is not cleared.")
    parser.add_argument("--record-warmup-before-lift", type=float, default=WARMUP_SECONDS,
                        help="Deprecated; kept for CLI compatibility. Memory-first recording starts when lift motion is detected.")
    parser.add_argument("--camera-warmup-seconds", type=float, default=CAMERA_WARMUP_SECONDS,
                        help="Drain OAK streams for this many seconds before starting robot motion.")
    parser.add_argument("--record-after-lift", type=float, default=1.0)
    parser.add_argument("--state-publish-hz", type=float, default=STATE_PUBLISH_HZ, help="Measured FCI state publish rate for recording; default 100 Hz.")
    parser.add_argument("--prelift-settle-s", type=float, default=0.30,
                        help="Extra settle time after blocking hover/descend before the next command.")
    parser.add_argument("--cartesian-runtime-joint-margin-rad", type=float, default=0.15,
                        help="Abort Cartesian motion before any joint gets closer than this to a configured soft limit.")
    parser.add_argument("--lift-speed", type=float, default=0.05,
                        help="Average Cartesian lift speed in m/s; independent of --motion-speed.")
    parser.add_argument("--motion-start-delta-m", type=float, default=0.0015,
                        help="Measured TCP displacement from its lift start position at which memory-first RGB-D recording starts (also the threshold for --motion-start-timeout-s).")
    parser.add_argument("--motion-start-timeout-s", type=float, default=3.0,
                        help="Abort lift if measured TCP does not move by --motion-start-delta-m within this time.")
    # Only --servo-max-angular is read: each direct-FCI move lasts at least rotation / this value,
    # capping the average angular speed (peak up to 1.875x); the other --servo-* flags have no effect.
    parser.add_argument("--servo-max-linear", type=float, default=MAX_LIN, help="No effect in this executor; kept for CLI/summary compatibility.")
    parser.add_argument("--servo-max-angular", type=float, default=MAX_ANG, help="Average angular speed cap in rad/s: each move lasts at least its rotation angle / this value (the minimum-jerk peak is up to 1.875x).")
    parser.add_argument("--servo-kp-linear", type=float, default=1.2, help="No effect in this executor; kept for CLI/summary compatibility.")
    parser.add_argument("--servo-kp-angular", type=float, default=1.2, help="No effect in this executor; kept for CLI/summary compatibility.")
    parser.add_argument("--servo-pos-tol", type=float, default=0.006, help="No effect in this executor; kept for CLI/summary compatibility.")
    parser.add_argument("--servo-rot-tol", type=float, default=0.035, help="No effect in this executor; kept for CLI/summary compatibility.")
    parser.add_argument("--servo-timeout", type=float, default=30.0, help="No effect in this executor; kept for CLI/summary compatibility.")

# =========================
# Main process
# =========================
def main() -> None:
    global FPS

    parser = grasp_exec.build_parser()
    add_recording_and_servo_args(parser)
    args = parser.parse_args()
    if args.cartesian_runtime_joint_margin_rad < 0.0:
        raise ValueError("--cartesian-runtime-joint-margin-rad must be non-negative")
    if args.lift_speed <= 0.0:
        raise ValueError("--lift-speed must be positive")
    if args.execute and not args.ip:
        parser.error("--execute needs the robot IP: set AURA_ROBOT_IP in config/aura.env and load it (set -a; source config/aura.env; set +a) or pass --ip <ROBOT_IP>")
    if args.execute and not args.record_oak_ids:
        parser.error("--execute needs the OAK MXIDs: set AURA_OAK_IDS in config/aura.env and load it (set -a; source config/aura.env; set +a) or pass --record-oak-ids <MXID> [<MXID> ...]")

    if args.record_output_root is None:
        args.record_output_root = infer_case_phystwin_output_root(Path(args.grasp_json))
        print(f"[config] inferred recording output from --grasp-json: {args.record_output_root}")
    else:
        args.record_output_root = Path(args.record_output_root).expanduser().resolve()
        print(f"[config] using explicit recording output: {args.record_output_root}")

    # Configure recording globals before child processes are spawned.
    FPS = int(args.record_fps)
    apply_record_resolution(args.record_resolution)
    print(f"[config] memory-first 3-OAK record_resolution={RECORD_RESOLUTION}, FPS={FPS}, OAK={OAK_W}x{OAK_H}")
    sources = [cfg.get("config_source", "") for cfg in HARDCODED_CAMERA_CONFIGS.values()]
    if any("scaled_from_low" in s for s in sources):
        print("[config][WARN] At least one metadata intrinsic is a scaled fallback; real calibration file was not found.")
    else:
        print("[config] camera metadata intrinsics loaded from matching calibration files when available.")

    calib_source_path = Path(args.calib).expanduser().resolve()
    robot_calibration_copy = None
    if args.execute:
        robot_calibration_copy = copy_robot_calibration_to_phystwin(
            calib_source_path,
            Path(args.record_output_root),
        )
        copied_calib_path = (
            Path(args.record_output_root)
            / str(robot_calibration_copy["path_relative_to_phystwin"])
        ).resolve()
        args.calib_source_original = str(calib_source_path)
        args.calib = str(copied_calib_path)
        print(f"[config] robot calibration copied: {copied_calib_path}")
    else:
        args.calib = str(calib_source_path)

    # In execute mode, load the copied file. This guarantees that the bytes
    # used for world-to-base transformation are exactly the bytes preserved
    # alongside the recording.
    calib = grasp_exec.load_calib(Path(args.calib))
    if robot_calibration_copy is not None:
        robot_calibration_copy["stored_keys"] = sorted(calib.keys())
        robot_calibration_copy["transforms_used"] = {
            key: np.asarray(calib[key], dtype=float).tolist()
            for key in [
                "T_base_from_world",
                "T_world_from_base",
                "T_world_from_camera",
                "T_base_from_camera",
                "T_ee_from_board",
                "T_board_from_ee",
            ]
            if key in calib
        }
    args.robot_calibration_copy = robot_calibration_copy

    T_input_from_grasp, rec = grasp_exec.load_grasp_pose(Path(args.grasp_json), args.rank)

    pose_frame = rec.get("_detected_frame", args.pose_frame)
    if rec.get("_detected_frame") == "base" and args.pose_frame != "base":
        print("[info] JSON contains base_pose_matrix_4x4; overriding pose_frame to base.")
        pose_frame = "base"

    T_base_from_grasp = grasp_exec.transform_pose_to_base(T_input_from_grasp, pose_frame, calib)
    T_base_from_target = grasp_exec.apply_offsets(
        T_base_from_grasp,
        args.tcp_offset_grasp_frame,
        args.rot_offset_rpy,
    )

    if args.orientation_mode == "home":
        if args.home_rpy is not None:
            T_base_from_target[:3, :3] = grasp_exec.rpy_to_matrix(*args.home_rpy)
        else:
            raise RuntimeError(
                "This two-process script requires --home-rpy when using --orientation-mode home. "
                "Use --orientation-mode anygrasp, or provide --home-rpy R P Y."
            )

    # This executor relies on the upstream table/PyRoki filter instead of the
    # legacy Cartesian x/y/z workspace box. Keep inherited CLI flags accepted
    # for compatibility, but the workspace gate is intentionally disabled.
    args.disable_workspace_check = True
    args.allow_outside_workspace = True
    print("[config] legacy Cartesian workspace limit disabled; table/PyRoki filtering remains enabled upstream.")
    rpy = grasp_exec.print_pose_summary(T_base_from_target, rec, args)
    x0, y0, z0 = [float(v) for v in T_base_from_target[:3, 3]]
    expected_hover_T = T_base_from_target.copy()
    expected_hover_T[2, 3] += float(args.hover_z_offset)
    _validate_pyroki_hover_target(rec, expected_hover_T)
    robustness = rec.get("table_ik_fk_check", {}).get("home_to_hover_cartesian_robustness")
    if not isinstance(robustness, dict) or not robustness.get("safe", False):
        raise RuntimeError(
            "Selected grasp lacks a passing dense Cartesian robustness report. "
            "Re-run the PyRoki filter (bash scripts/anygrasp_poses_one_shot.sh --resume --refilter <case>) before execution."
        )
    print(
        "[config] dense Cartesian robustness passed: "
        f"rotation={float(robustness['cartesian_rotation_deg']):.1f}deg, "
        f"endpoint_max_joint_delta={float(robustness['endpoint_max_joint_delta_rad']):.3f}rad, "
        f"minimum_soft_margin={float(robustness['minimum_soft_joint_margin_rad']):.3f}rad"
    )

    target_pose = [x0, y0, z0, float(rpy[0]), float(rpy[1]), float(rpy[2])]

    if not args.execute:
        print("\n[dry-run] Robot was NOT connected. Add --execute --home to move the robot and record lift.")
        return

    ctx = mp.get_context("spawn")
    record_evt = ctx.Event()
    stop_evt = ctx.Event()
    camera_ready_evt = ctx.Event()
    status_q: mp.Queue = ctx.Queue()
    console_lock = ctx.Lock()
    shared_robot_state = create_shared_robot_state(ctx)

    args_dict = vars(args).copy()
    # Make Path-like values spawn/json friendly.
    for key, value in list(args_dict.items()):
        if isinstance(value, Path):
            args_dict[key] = str(value)

    recorder_proc = ctx.Process(
        target=recorder_process_main,
        args=(
            args,
            record_evt,
            stop_evt,
            status_q,
            shared_robot_state,
            camera_ready_evt,
        ),
        name="recorder_process",
    )
    robot_proc = ctx.Process(
        target=anygrasp_robot_process_main,
        args=(
            record_evt, stop_evt, status_q, shared_robot_state,
            args_dict, target_pose, rec, camera_ready_evt, console_lock,
        ),
        name="anygrasp_robot_process",
    )

    # Start recorder first so cameras are ready while user/robot stage proceeds.
    recorder_proc.start()
    robot_proc.start()

    interrupted = False
    try:
        while True:
            try:
                msg = status_q.get(timeout=MAIN_STATUS_POLL_TIMEOUT_S)
                with console_lock:
                    print(msg, flush=True)
            except queue.Empty:
                pass

            if stop_evt.is_set():
                break

            if (not recorder_proc.is_alive()) and (not robot_proc.is_alive()):
                break

            if recorder_proc.exitcode not in (None, 0):
                stop_evt.set()
                break
            if robot_proc.exitcode not in (None, 0):
                stop_evt.set()
                break

    except KeyboardInterrupt:
        interrupted = True
        stop_evt.set()
    finally:
        stop_evt.set()
        record_evt.clear()

        recorder_proc.join(timeout=10.0)
        robot_proc.join(timeout=10.0)

        if recorder_proc.is_alive():
            recorder_proc.terminate()
        if robot_proc.is_alive():
            robot_proc.terminate()

        recorder_proc.join(timeout=2.0)
        robot_proc.join(timeout=2.0)

        while True:
            try:
                msg = status_q.get_nowait()
                with console_lock:
                    print(msg, flush=True)
            except queue.Empty:
                break

        robot_exitcode = robot_proc.exitcode
        recorder_exitcode = recorder_proc.exitcode
        print(
            f"[process] robot_exitcode={robot_exitcode} "
            f"recorder_exitcode={recorder_exitcode}"
        )

    if interrupted:
        raise KeyboardInterrupt
    failures = []
    if robot_exitcode != 0:
        failures.append(f"robot process exit code {robot_exitcode}")
    if recorder_exitcode != 0:
        failures.append(f"recorder process exit code {recorder_exitcode}")
    if failures:
        raise RuntimeError("Execution failed: " + "; ".join(failures))
    print("Done.")


if __name__ == "__main__":
    main()
