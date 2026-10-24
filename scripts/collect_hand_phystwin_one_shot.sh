#!/usr/bin/env bash
set -Eeuo pipefail

_aura_self="${BASH_SOURCE[0]}"; [[ ! -L "$_aura_self" ]] || _aura_self="$(readlink -f -- "$_aura_self")"; AURA_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$_aura_self")/.." && pwd)"
if [[ -f "$AURA_REPO_ROOT/config/aura.env" ]]; then
  source "$AURA_REPO_ROOT/config/aura.env"
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/collect_hand_phystwin_one_shot.sh [options] CASE_NAME

Interactive workflow:
  1. Create $AURA_DATA_ROOT/hand_collected/CASE_NAME (or --data-root/CASE_NAME)
  2. Warm up and start synchronized 3-OAK RGB-D recording
  3. Press Enter to stop (the recorder also stops on its own after --max-duration, default 30 s)
  4. Generate a numbered 3-camera preview video
  5. Ask for inclusive start/end frame indices unless supplied on the CLI
  6. Convert the selected range to PhysTwin color/depth layout
  7. Generate calibrate.pkl, metadata.json, calibration archive, and MP4 previews

Options:
  --overwrite                       Delete and recreate the entire case directory
  --start-frame N                   First synchronized bundle to keep; default: prompt
  --end-frame N                     Last synchronized bundle to keep, INCLUSIVE; default: prompt
  --max-duration SEC                Recorder safety limit; default: 30, maximum: 30
  --resolution low|high             Default: high
  --fps N                           Output/sensor/sync FPS; default: 15
  --sync-tolerance-ms MS            Dynamic synchronization tolerance; default: 45
  --camera-warmup-seconds SEC       Default: 4
  --preview / --no-preview          Live OAK RGB preview; default: enabled
  --make-mp4 / --no-mp4             Generate final per-camera MP4 previews; default: enabled
  --aura-root PATH                  Repository root for the aura/ scripts; default: parent of scripts/
                                    (AURA_* paths already set by config/aura.env or the shell do not move with it)
  --aura-python PATH                Default: python of the conda env $AURA_ENV (AuRA)
  --data-root PATH                  Default: $AURA_DATA_ROOT/hand_collected
  --record-script PATH              Override aura/capture/record_3oak.py
  --convert-script PATH             Override aura/conversion/raw_data_to_phystwin_style.py
  --calibration-script PATH         Override aura/conversion/calibration_matrix_to_phystwin_style.py
  -h, --help

Configuration (config/aura.env, see config/aura.env.example):
  AURA_OAK_IDS     Required: OAK MXIDs, space separated, in camera-index order 0 1 2
  AURA_ENV         Conda env that runs every stage (or set AURA_PYTHON)
  AURA_DATA_ROOT   Default: data/ in the repository
  AURA_CALIB_ROOT  Default: calibration/ in the repository; must contain
                   intrinsic_results/1280x720/oak_<MXID>.{json,npz} (640x400/ for --resolution low)
                   and extrinsic_results/oak_<MXID>.{json,npz}
Most options also read an environment variable of their own (DATA_ROOT, RESOLUTION,
FPS, SYNC_TOLERANCE_MS, OVERWRITE, ...), which takes precedence over config/aura.env.
--start-frame, --end-frame and --aura-root are command-line only.

The final case directory itself is PhysTwin-style:
  $AURA_DATA_ROOT/hand_collected/CASE_NAME/ (or --data-root/CASE_NAME/)
    color/{0,1,2}/<frame>.png
    depth/{0,1,2}/<frame>.npy
    calibrate.pkl
    metadata.json
    raw_recording/bundle_******/...
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

need_value() {
  [[ $# -ge 2 ]] || die "Option $1 requires a value"
}

run_stage() {
  local stage_name="$1"
  shift
  echo
  echo "================================================================================"
  echo "[STAGE] $stage_name"
  echo "================================================================================"
  "$@"
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_positive_number() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] &&
    awk -v value="$1" 'BEGIN { exit !(value > 0) }'
}

AURA_ENV="${AURA_ENV:-AuRA}"
AURA_PYTHON="${AURA_PYTHON:-}"
DATA_ROOT="${DATA_ROOT:-}"
RECORD_SCRIPT="${RECORD_SCRIPT:-}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-}"
CALIBRATION_SCRIPT="${CALIBRATION_SCRIPT:-}"

RESOLUTION="${RESOLUTION:-high}"
FPS="${FPS:-15}"
SYNC_TOLERANCE_MS="${SYNC_TOLERANCE_MS:-45}"
CAMERA_WARMUP_SECONDS="${CAMERA_WARMUP_SECONDS:-4}"
MAX_DURATION="${MAX_DURATION:-30}"
LIVE_PREVIEW="${LIVE_PREVIEW:-True}"
MAKE_MP4="${MAKE_MP4:-1}"
OVERWRITE="${OVERWRITE:-0}"
START_FRAME=""
END_FRAME=""
CASE_NAME=""

read -r -d '' -a OAK_IDS <<< "${AURA_OAK_IDS:-}" || true
CAMERA_ORDER=("${OAK_IDS[@]/#/oak_}")

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --start-frame) need_value "$@"; START_FRAME="$2"; shift 2 ;;
    --end-frame) need_value "$@"; END_FRAME="$2"; shift 2 ;;
    --max-duration) need_value "$@"; MAX_DURATION="$2"; shift 2 ;;
    --resolution) need_value "$@"; RESOLUTION="$2"; shift 2 ;;
    --fps) need_value "$@"; FPS="$2"; shift 2 ;;
    --sync-tolerance-ms) need_value "$@"; SYNC_TOLERANCE_MS="$2"; shift 2 ;;
    --camera-warmup-seconds) need_value "$@"; CAMERA_WARMUP_SECONDS="$2"; shift 2 ;;
    --preview) LIVE_PREVIEW=True; shift ;;
    --no-preview) LIVE_PREVIEW=False; shift ;;
    --make-mp4) MAKE_MP4=1; shift ;;
    --no-mp4) MAKE_MP4=0; shift ;;
    --aura-root) need_value "$@"; AURA_REPO_ROOT="$2"; shift 2 ;;
    --aura-python) need_value "$@"; AURA_PYTHON="$2"; shift 2 ;;
    --data-root) need_value "$@"; DATA_ROOT="$2"; shift 2 ;;
    --record-script) need_value "$@"; RECORD_SCRIPT="$2"; shift 2 ;;
    --convert-script) need_value "$@"; CONVERT_SCRIPT="$2"; shift 2 ;;
    --calibration-script) need_value "$@"; CALIBRATION_SCRIPT="$2"; shift 2 ;;
    --*) die "Unknown option: $1" ;;
    *)
      [[ -z "$CASE_NAME" ]] || die "Only one CASE_NAME is allowed"
      CASE_NAME="$1"
      shift ;;
  esac
done

[[ -n "$CASE_NAME" ]] || { usage >&2; exit 2; }
[[ "$CASE_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || die "Invalid CASE_NAME: $CASE_NAME"
[[ "$RESOLUTION" == "low" || "$RESOLUTION" == "high" ]] || die "--resolution must be low or high"
is_nonnegative_integer "$FPS" && [[ "$FPS" -gt 0 ]] || die "--fps must be a positive integer"
is_positive_number "$MAX_DURATION" || die "--max-duration must be positive"
awk -v value="$MAX_DURATION" 'BEGIN { exit !(value <= 30) }' || die "--max-duration cannot exceed record_3oak.py's 30-second safety cap"
is_positive_number "$SYNC_TOLERANCE_MS" || die "--sync-tolerance-ms must be positive"
is_positive_number "$CAMERA_WARMUP_SECONDS" || die "--camera-warmup-seconds must be positive"
[[ -z "$START_FRAME" ]] || is_nonnegative_integer "$START_FRAME" || die "--start-frame must be a non-negative integer"
[[ -z "$END_FRAME" ]] || is_nonnegative_integer "$END_FRAME" || die "--end-frame must be a non-negative integer"
[[ "$OVERWRITE" == "0" || "$OVERWRITE" == "1" ]] || die "OVERWRITE must be 0 or 1"
[[ "${#OAK_IDS[@]}" -gt 0 ]] || die "AURA_OAK_IDS is empty: set the OAK MXIDs (space separated, camera-index order) in config/aura.env"

if [[ -z "$AURA_PYTHON" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$AURA_ENV" && -n "${CONDA_PREFIX:-}" ]]; then
    AURA_PYTHON="$CONDA_PREFIX/bin/python"
  else
    command -v conda >/dev/null || die "conda not found: activate the '$AURA_ENV' environment or pass --aura-python"
    AURA_PYTHON="$(conda run -n "$AURA_ENV" python -c 'import sys; print(sys.executable)' 2>/dev/null)" || die "conda cannot find the env '$AURA_ENV': activate it or pass --aura-python"
  fi
fi
AURA_DATA_ROOT="${AURA_DATA_ROOT:-$AURA_REPO_ROOT/data}"
AURA_CALIB_ROOT="${AURA_CALIB_ROOT:-$AURA_REPO_ROOT/calibration}"
[[ -n "$DATA_ROOT" ]] || DATA_ROOT="$AURA_DATA_ROOT/hand_collected"
[[ -n "$RECORD_SCRIPT" ]] || RECORD_SCRIPT="$AURA_REPO_ROOT/aura/capture/record_3oak.py"
[[ -n "$CONVERT_SCRIPT" ]] || CONVERT_SCRIPT="$AURA_REPO_ROOT/aura/conversion/raw_data_to_phystwin_style.py"
[[ -n "$CALIBRATION_SCRIPT" ]] || CALIBRATION_SCRIPT="$AURA_REPO_ROOT/aura/conversion/calibration_matrix_to_phystwin_style.py"

[[ -x "$AURA_PYTHON" ]] || die "AuRA Python not executable: $AURA_PYTHON"
[[ -f "$RECORD_SCRIPT" ]] || die "3-OAK recorder not found: $RECORD_SCRIPT"
[[ -f "$CONVERT_SCRIPT" ]] || die "Raw-to-PhysTwin converter not found: $CONVERT_SCRIPT"
[[ -f "$CALIBRATION_SCRIPT" ]] || die "Calibration converter not found: $CALIBRATION_SCRIPT"

if [[ "$RESOLUTION" == "high" ]]; then
  INTRINSIC_DIR="$AURA_CALIB_ROOT/intrinsic_results/1280x720"
else
  INTRINSIC_DIR="$AURA_CALIB_ROOT/intrinsic_results/640x400"
fi
EXTRINSIC_DIR="$AURA_CALIB_ROOT/extrinsic_results"
[[ -d "$INTRINSIC_DIR" ]] || die "Intrinsic calibration directory not found: $INTRINSIC_DIR"
[[ -d "$EXTRINSIC_DIR" ]] || die "Extrinsic calibration directory not found: $EXTRINSIC_DIR"

DATA_ROOT_CANONICAL="$(realpath -m -- "$DATA_ROOT")"
CASE_ROOT="$(realpath -m -- "$DATA_ROOT/$CASE_NAME")"
EXPECTED_CASE_ROOT="$DATA_ROOT_CANONICAL/$CASE_NAME"
[[ "$CASE_ROOT" == "$EXPECTED_CASE_ROOT" ]] || die "Resolved case path is unexpected: $CASE_ROOT"
[[ "$CASE_ROOT" != "/" && "$CASE_ROOT" != "$DATA_ROOT_CANONICAL" ]] || die "Unsafe case path: $CASE_ROOT"

if [[ -e "$CASE_ROOT" ]]; then
  if [[ "$OVERWRITE" != "1" ]]; then
    die "Case already exists: $CASE_ROOT. Choose another name or pass --overwrite."
  fi
  echo "[OVERWRITE] Deleting entire existing hand-collected case: $CASE_ROOT"
  rm -rf -- "$CASE_ROOT"
  [[ ! -e "$CASE_ROOT" ]] || die "Failed to remove existing case: $CASE_ROOT"
fi

RAW_ROOT="$CASE_ROOT/raw_recording"
RECORD_LOG="$CASE_ROOT/recording_console.log"
RAW_PREVIEW="$CASE_ROOT/raw_preview_3oak_numbered.mp4"
mkdir -p "$RAW_ROOT"
touch "$RECORD_LOG"

echo "[case] output       = $CASE_ROOT"
echo "[case] raw bundles  = $RAW_ROOT"
echo "[config] resolution = $RESOLUTION"
echo "[config] fps        = $FPS"
echo "[config] sync tol   = ${SYNC_TOLERANCE_MS} ms"
echo "[config] max time   = ${MAX_DURATION} s"

REC_PID=""
abort_handler() {
  trap - INT TERM
  echo
  echo "[interrupt] Stopping recorder safely..." >&2
  if [[ -n "$REC_PID" ]] && kill -0 "$REC_PID" 2>/dev/null; then
    kill -INT "$REC_PID" 2>/dev/null || true
    wait "$REC_PID" 2>/dev/null || true
  fi
  exit 130
}
trap abort_handler INT TERM

echo
echo "================================================================================"
echo "[STAGE] 01_record_hand_manipulation"
echo "================================================================================"

# Bash may mark asynchronous children as ignoring SIGINT. Reset the signal
# disposition in the child shell before exec so Python installs its normal
# KeyboardInterrupt handler and record_3oak.py can synchronize/save on Enter.
(
  trap - INT TERM
  exec env PYTHONUNBUFFERED=1 "$AURA_PYTHON" "$RECORD_SCRIPT" \
    --oak-ids "${OAK_IDS[@]}" \
    --resolution "$RESOLUTION" \
    --fps "$FPS" \
    --sensor-fps "$FPS" \
    --sync-fps "$FPS" \
    --output-root "$RAW_ROOT" \
    --warmup-seconds "$CAMERA_WARMUP_SECONDS" \
    --duration "$MAX_DURATION" \
    --preview "$LIVE_PREVIEW" \
    --sync-tolerance-ms "$SYNC_TOLERANCE_MS" \
    --oak-internal-sync-ms 25 \
    --enable-undistortion True \
    --overwrite
) > >(tee -a "$RECORD_LOG") 2>&1 &
REC_PID=$!

RECORDING_STARTED=0
for ((attempt = 0; attempt < 1200; attempt++)); do
  if grep -Fq "Start MEMORY recording" "$RECORD_LOG"; then
    RECORDING_STARTED=1
    break
  fi
  if ! kill -0 "$REC_PID" 2>/dev/null; then
    set +e
    wait "$REC_PID"
    recorder_code=$?
    set -e
    die "Recorder exited before recording started (exit=$recorder_code). See $RECORD_LOG"
  fi
  sleep 0.1
done

if [[ "$RECORDING_STARTED" != "1" ]]; then
  kill -INT "$REC_PID" 2>/dev/null || true
  wait "$REC_PID" 2>/dev/null || true
  die "Timed out waiting for OAK recording to start. See $RECORD_LOG"
fi

echo
echo "[READY] Recording is active. Keep the first frame clear, then manipulate the object."
echo "[READY] Press Enter here to stop. It will also stop automatically after ${MAX_DURATION}s."

stop_requested=0
while kill -0 "$REC_PID" 2>/dev/null; do
  if read -r -t 0.25 _unused; then
    stop_requested=1
    break
  fi
done

if [[ "$stop_requested" == "1" ]] && kill -0 "$REC_PID" 2>/dev/null; then
  echo "[record] Enter received; requesting a safe recorder stop..."
  kill -INT "$REC_PID"
fi

echo "[record] Waiting for offline synchronization and disk save..."
set +e
wait "$REC_PID"
recorder_code=$?
set -e
REC_PID=""
[[ "$recorder_code" -eq 0 ]] || die "Recorder failed with exit code $recorder_code. See $RECORD_LOG"
[[ -f "$RAW_ROOT/recording_summary.json" ]] || die "Recorder did not create recording_summary.json"

mapfile -t BUNDLES < <(find "$RAW_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'bundle_*' -printf '%f\n' | sort)
NUM_BUNDLES="${#BUNDLES[@]}"
[[ "$NUM_BUNDLES" -ge 2 ]] || die "Only $NUM_BUNDLES synchronized bundle(s) were saved; record a longer sequence"
LAST_BUNDLE_INDEX=$((NUM_BUNDLES - 1))
echo "[record] synchronized bundles: $NUM_BUNDLES (valid indices: 0..$LAST_BUNDLE_INDEX)"

run_stage "02_generate_numbered_raw_preview" \
  "$AURA_PYTHON" - "$RAW_ROOT" "$RAW_PREVIEW" "$FPS" "${CAMERA_ORDER[@]}" <<'PY'
import sys
from pathlib import Path

import cv2

raw_root = Path(sys.argv[1])
output = Path(sys.argv[2])
fps = float(sys.argv[3])
cameras = sys.argv[4:]
bundles = sorted(p for p in raw_root.glob("bundle_*") if p.is_dir())
if not bundles:
    raise RuntimeError(f"No bundles found: {raw_root}")

panels0 = []
for camera in cameras:
    image = cv2.imread(str(bundles[0] / camera / "rgb.jpg"), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read camera {camera} from {bundles[0]}")
    height, width = image.shape[:2]
    preview_width = 480
    preview_height = int(round(height * preview_width / width))
    panels0.append(cv2.resize(image, (preview_width, preview_height)))

frame_height = panels0[0].shape[0]
frame_width = sum(panel.shape[1] for panel in panels0)
temporary = output.with_name(f".{output.stem}.tmp.mp4")
temporary.unlink(missing_ok=True)
writer = None
selected_codec = None
for codec in ("mp4v", "FMP4", "MJPG"):
    candidate = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*codec), fps, (frame_width, frame_height)
    )
    if candidate.isOpened():
        writer = candidate
        selected_codec = codec
        break
    candidate.release()
if writer is None:
    raise RuntimeError(f"Cannot create MP4 preview: {temporary}")

try:
    for frame_index, bundle in enumerate(bundles):
        panels = []
        for cam_idx, camera in enumerate(cameras):
            image = cv2.imread(str(bundle / camera / "rgb.jpg"), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot read {bundle / camera / 'rgb.jpg'}")
            image = cv2.resize(image, (panels0[cam_idx].shape[1], frame_height))
            cv2.putText(image, f"cam{cam_idx}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"frame {frame_index}", (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            panels.append(image)
        writer.write(cv2.hconcat(panels))
finally:
    writer.release()

if not temporary.is_file() or temporary.stat().st_size == 0:
    raise RuntimeError(f"Preview writer produced no data: {temporary}")
temporary.replace(output)
print(f"Saved numbered 3-OAK preview: {output} (frames={len(bundles)}, codec={selected_codec})")
PY

echo
echo "Inspect the numbered preview before choosing frames:"
echo "  $RAW_PREVIEW"
echo "Frame selection uses synchronized bundle indices 0..$LAST_BUNDLE_INDEX."
echo "The end frame entered below is INCLUSIVE."

while true; do
  if [[ -z "$START_FRAME" ]]; then
    read -r -p "Start frame [0]: " input_start
    input_start="${input_start:-0}"
  else
    input_start="$START_FRAME"
  fi
  if is_nonnegative_integer "$input_start" && [[ "$input_start" -le "$LAST_BUNDLE_INDEX" ]]; then
    START_FRAME="$input_start"
    break
  fi
  echo "Invalid start frame. Enter an integer from 0 to $LAST_BUNDLE_INDEX."
  START_FRAME=""
done

while true; do
  if [[ -z "$END_FRAME" ]]; then
    read -r -p "End frame, inclusive [$LAST_BUNDLE_INDEX]: " input_end
    input_end="${input_end:-$LAST_BUNDLE_INDEX}"
  else
    input_end="$END_FRAME"
  fi
  if is_nonnegative_integer "$input_end" && \
     [[ "$input_end" -ge "$START_FRAME" ]] && \
     [[ "$input_end" -le "$LAST_BUNDLE_INDEX" ]]; then
    END_FRAME="$input_end"
    break
  fi
  echo "Invalid end frame. Enter an integer from $START_FRAME to $LAST_BUNDLE_INDEX."
  END_FRAME=""
done

END_FRAME_EXCLUSIVE=$((END_FRAME + 1))
SELECTED_FRAME_COUNT=$((END_FRAME - START_FRAME + 1))
echo "[selection] keeping inclusive frames $START_FRAME..$END_FRAME ($SELECTED_FRAME_COUNT frames)"

run_stage "03_convert_selected_frames_to_phystwin" \
  "$AURA_PYTHON" "$CONVERT_SCRIPT" \
    --raw-root "$RAW_ROOT" \
    --output-root "$CASE_ROOT" \
    --camera-order "${CAMERA_ORDER[@]}" \
    --start-frame "$START_FRAME" \
    --end-frame "$END_FRAME_EXCLUSIVE" \
    --stride 1 \
    --copy-meta \
    --overwrite

[[ -f "$CASE_ROOT/metadata.json" ]] || die "Conversion did not create metadata.json"
cp --preserve=timestamps "$CASE_ROOT/metadata.json" "$CASE_ROOT/conversion_metadata.json"

FRAME_COUNT="$(find "$CASE_ROOT/color/0" -maxdepth 1 -type f -name '*.png' | wc -l)"
[[ "$FRAME_COUNT" -eq "$SELECTED_FRAME_COUNT" ]] || \
  die "Converted frame count $FRAME_COUNT does not match selected count $SELECTED_FRAME_COUNT"

run_stage "04_generate_phystwin_calibration" \
  "$AURA_PYTHON" "$CALIBRATION_SCRIPT" \
    --intrinsic-dir "$INTRINSIC_DIR" \
    --extrinsic-dir "$EXTRINSIC_DIR" \
    --output-dir "$CASE_ROOT" \
    --camera-order "${CAMERA_ORDER[@]}" \
    --serial-numbers "${CAMERA_ORDER[@]}" \
    --frame-num "$FRAME_COUNT" \
    --force-single-wh \
    --overwrite

[[ -f "$CASE_ROOT/calibrate.pkl" ]] || die "Calibration stage did not create calibrate.pkl"
[[ -f "$CASE_ROOT/metadata.json" ]] || die "Calibration stage did not create metadata.json"

echo
echo "================================================================================"
echo "[STAGE] 04b_archive_camera_calibration"
echo "================================================================================"
mkdir -p "$CASE_ROOT/camera_calibration/intrinsic" "$CASE_ROOT/camera_calibration/extrinsic"
for camera in "${CAMERA_ORDER[@]}"; do
  for extension in json npz; do
    if [[ -f "$INTRINSIC_DIR/$camera.$extension" ]]; then
      cp --preserve=timestamps "$INTRINSIC_DIR/$camera.$extension" "$CASE_ROOT/camera_calibration/intrinsic/"
    fi
    if [[ -f "$EXTRINSIC_DIR/$camera.$extension" ]]; then
      cp --preserve=timestamps "$EXTRINSIC_DIR/$camera.$extension" "$CASE_ROOT/camera_calibration/extrinsic/"
    fi
  done
done

if [[ "$MAKE_MP4" == "1" ]]; then
  run_stage "05_generate_final_camera_mp4_previews" \
    "$AURA_PYTHON" - "$CASE_ROOT" "$FPS" <<'PY'
import sys
from pathlib import Path

import cv2

root = Path(sys.argv[1])
fps = float(sys.argv[2])
for camera_index in range(3):
    frame_dir = root / "color" / str(camera_index)
    frames = sorted(frame_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not frames:
        raise RuntimeError(f"No PNG frames for camera {camera_index}: {frame_dir}")
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frames[0]}")
    height, width = first.shape[:2]
    output = root / "color" / f"{camera_index}.mp4"
    temporary = root / "color" / f".{camera_index}.tmp.mp4"
    temporary.unlink(missing_ok=True)
    writer = None
    selected_codec = None
    for codec in ("mp4v", "FMP4", "MJPG"):
        candidate = cv2.VideoWriter(
            str(temporary), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if candidate.isOpened():
            writer = candidate
            selected_codec = codec
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"Cannot create MP4 for camera {camera_index}")
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot read frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(f"Frame size changed: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"MP4 writer produced no data: {temporary}")
    temporary.replace(output)
    print(f"Saved {output} ({len(frames)} frames, codec={selected_codec}, fps={fps:g})")
PY
fi

export CASE_NAME CASE_ROOT RAW_ROOT RAW_PREVIEW START_FRAME END_FRAME FRAME_COUNT
export RESOLUTION FPS SYNC_TOLERANCE_MS CAMERA_WARMUP_SECONDS MAX_DURATION
"$AURA_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

case_root = Path(os.environ["CASE_ROOT"])
recording_summary_path = Path(os.environ["RAW_ROOT"]) / "recording_summary.json"
recording_summary = json.load(open(recording_summary_path, "r", encoding="utf-8"))
summary = {
    "case_name": os.environ["CASE_NAME"],
    "case_root": str(case_root),
    "raw_recording": os.environ["RAW_ROOT"],
    "numbered_raw_preview": os.environ["RAW_PREVIEW"],
    "selected_start_frame_inclusive": int(os.environ["START_FRAME"]),
    "selected_end_frame_inclusive": int(os.environ["END_FRAME"]),
    "selected_frame_count": int(os.environ["FRAME_COUNT"]),
    "resolution": os.environ["RESOLUTION"],
    "fps": int(os.environ["FPS"]),
    "sync_tolerance_ms": float(os.environ["SYNC_TOLERANCE_MS"]),
    "camera_warmup_seconds": float(os.environ["CAMERA_WARMUP_SECONDS"]),
    "requested_max_duration_seconds": float(os.environ["MAX_DURATION"]),
    "offline_sync": recording_summary.get("offline_sync"),
    "phys_twin_required_files": {
        "color": str(case_root / "color"),
        "depth": str(case_root / "depth"),
        "calibrate_pkl": str(case_root / "calibrate.pkl"),
        "metadata_json": str(case_root / "metadata.json"),
    },
}
with open(case_root / "hand_collection_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
print(json.dumps(summary, indent=2))
PY

echo
echo "================================================================================"
echo "[DONE] Hand-manipulation 3-OAK PhysTwin collection complete"
echo "Case:       $CASE_ROOT"
echo "Frames:     $FRAME_COUNT (raw indices $START_FRAME..$END_FRAME inclusive)"
echo "Raw data:   $RAW_ROOT"
echo "Preview:    $RAW_PREVIEW"
echo "Color:      $CASE_ROOT/color"
echo "Depth:      $CASE_ROOT/depth"
echo "Calibration: $CASE_ROOT/calibrate.pkl"
echo "Metadata:   $CASE_ROOT/metadata.json"
echo
echo "To use this case with the PhysTwin repository, link it into your PhysTwin clone:"
echo "  ln -s '$CASE_ROOT' '<PhysTwin clone>/data/different_types/$CASE_NAME'"
echo "================================================================================"
