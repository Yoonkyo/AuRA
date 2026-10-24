#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare one AuRA case through offline PyRoki filtering. This script never
# commands robot motion. Only CASE_NAME is required; every other option has a
# tested default and can be overridden on the command line or by environment
# (except --aura-root and --rot-offset-rpy, which are command-line only).
# Machine-specific settings (camera MXIDs, data/calibration/checkpoint/SDK
# locations, conda env names) come from config/aura.env.

_aura_self="${BASH_SOURCE[0]}"; [[ ! -L "$_aura_self" ]] || _aura_self="$(readlink -f -- "$_aura_self")"; AURA_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$_aura_self")/.." && pwd)"
if [[ -f "$AURA_REPO_ROOT/config/aura.env" ]]; then
  # shellcheck disable=SC1091
  source "$AURA_REPO_ROOT/config/aura.env"
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/anygrasp_poses_one_shot.sh [options] CASE_NAME

Pipeline (stages as named in <case>/logs/):
  01  Synchronized static 3-OAK RGB-D capture
      + PhysTwin-style static frame + camera calibration
  02  SAM/GroundingDINO segmentation and observation point-cloud filtering
  03  AnyGrasp pose generation, with optional no-collision fallback (03b)
  04  Offline table/fingertip/PyRoki IK-FK filtering (no robot motion)

General:
  --force                         Recapture and rebuild (keeps phystwin/); never on a case with a lift recording
  --resume                        Skip each stage whose own outputs exist, even if an earlier stage runs again
  --refilter                      With --resume: re-run PyRoki filtering even when safe JSON already exists
  --aura-root PATH                Repository root for the aura/ tools and assets/; default: parent of scripts/
                                  (AURA_* paths already set by config/aura.env or the shell do not move with it)
  --data-root PATH                Default: $AURA_DATA_ROOT/franka_lift
  --aura-env NAME                 Default: AuRA
  --anygrasp-env NAME             Default: Anygrasp

Static capture:
  --record-resolution low|high    Default: high
  --record-fps N                  Default: 15
  --record-sensor-fps N           Default: 15
  --record-sync-fps N             Default: 15
  --static-sync-tolerance-ms MS   Default: 70 (static scene)
  --camera-warmup-seconds SEC     Default: 4
  --record-duration SEC           Default: 1
  --preview / --no-preview        Default: no preview

Observation filtering (values used for AuRA data collection):
  --mask-object-id ID             Default: 0 (largest component)
  --voxel-size M                  Default: 0.003
  --stat-nb-neighbors N           Default: 35
  --stat-std-ratio R              Default: 1.2
  --radius-nb-points N            Default: 10
  --radius M                      Default: 0.006
  --pca-body-low P                Default: 1
  --pca-body-high P               Default: 99
  --pca-margin-long M             Default: 0
  --pca-margin-cross M            Default: 0
  --table-z M                     World z of the table; also the table plane of PyRoki filtering; default: 0
  --table-eps M                   Default: 0.01
  --sam-prompt TEXT               Default: inferred from CASE_NAME

AnyGrasp:
  --top-k N                       Default: 20
  --max-gripper-width M           Default: 0.08
  --top-down / --no-top-down      Default: top-down
  --collision-fallback / --no-collision-fallback
                                  Default: enabled
  --anygrasp-checkpoint PATH      Default: $AURA_ANYGRASP_SDK/grasp_detection/checkpoint_detection.tar
  --omp-num-threads N             Default: 12

PyRoki filtering:
  --pyroki-filter / --no-pyroki-filter
                                  Default: enabled
  --filter-script PATH            Default: aura/grasping/filter_grasps_table_collision_pyroki.py
  --robot-calib-results PATH      Calibration directory/NPZ/ZIP
                                  Default: $AURA_CALIB_ROOT/robot_calibration_results
  --state-file PATH               FR3 start-state NPZ; default: assets/fr3_home_state.npz
  --description NAME              Default: fr3_description
  --target-link NAME              Default: fr3_hand_tcp
  --pose-frame world|base|camera  Default: world
  --rot-offset-rpy R P Y          Grasp rotation offset (rad); default: 0 1.57079632679 0
  --offset-mode post|pre|none     Default: post
  --clearance-m M                 Default: 0.003
  --hover-z-offset M              Default: 0.10
  --final-z-offset M              Default: 0
  --lift-z-offset M               Default: 0.15
  --path-samples N                Default: 25
  --hover-path-linear-step-m M    Default: 0.02
  --hover-path-angular-step-deg D Default: 10
  --max-waypoint-joint-delta-rad RAD
                                  Default: 0.25
  --robust-joint-margin-rad RAD   Default: 0.16
  --max-home-hover-joint-delta-rad RAD
                                  Default: 2.0
  --max-home-hover-total-joint-motion-rad RAD
                                  Default: 8.0
  --max-home-hover-rotation-deg D Default: 120
  --camera-extrinsic-dir PATH     Default: $AURA_CALIB_ROOT/extrinsic_results
  --camera-keepout-radius-m M     Default: 0.12
  --max-position-error-m M        Default: 0.001
  --max-orientation-error-deg D   Default: 0.5
  --soft-limits / --no-soft-limits
                                  Default: enabled

Configuration (config/aura.env is sourced automatically; values already exported
in the shell take precedence; default paths are relative to the repository):
  AURA_OAK_IDS                    OAK MXIDs, space-separated, in camera-index order;
                                  required whenever the capture stage runs
  AURA_DATA_ROOT                  Default: data/ in the repository
  AURA_CALIB_ROOT                 Default: calibration/ in the repository
  AURA_GSAM_CHECKPOINTS           Grounded-SAM-2 weights; default: checkpoints/groundedSAM
  AURA_GSAM_REPO                  Your Grounded-SAM-2 clone (stage 02); default: third_party/Grounded-SAM-2
  AURA_ANYGRASP_SDK               Your AnyGrasp SDK checkout; default: third_party/anygrasp_sdk
  AURA_ENV / ANYGRASP_ENV         conda env names (same as --aura-env / --anygrasp-env)
  OAK_INTERNAL_SYNC_MS            On-device RGB-depth sync threshold; env only, default: 25
Most options also read an environment variable of their own (FORCE, RESUME, DATA_ROOT, TOP_K, OMP_NUM_THREADS,
...), which takes precedence over config/aura.env and replaces the default shown above. On/off variables
count as on only when set to 1 (PREVIEW: True or False). --aura-root and --rot-offset-rpy are command-line only.
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

need_value() {
  [[ $# -ge 2 ]] || die "Option $1 requires a value"
}

first_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

has_grasps() {
  local file="$1"
  python - "$file" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    print("1" if isinstance(value, list) and value else "0")
except Exception:
    print("0")
PY
}

# ------------------------- defaults -------------------------
DATA_ROOT="${DATA_ROOT:-}"
AURA_ENV="${AURA_ENV:-AuRA}"
ANYGRASP_ENV="${ANYGRASP_ENV:-Anygrasp}"

RECORD_RESOLUTION="${RECORD_RESOLUTION:-high}"
RECORD_FPS="${RECORD_FPS:-15}"
RECORD_SENSOR_FPS="${RECORD_SENSOR_FPS:-15}"
RECORD_SYNC_FPS="${RECORD_SYNC_FPS:-15}"
RECORD_DURATION="${RECORD_DURATION:-1}"
CAMERA_WARMUP_SECONDS="${CAMERA_WARMUP_SECONDS:-4}"
STATIC_SYNC_TOLERANCE_MS="${STATIC_SYNC_TOLERANCE_MS:-${SYNC_TOLERANCE_MS:-70}}"
OAK_INTERNAL_SYNC_MS="${OAK_INTERNAL_SYNC_MS:-25}"
PREVIEW="${PREVIEW:-False}"

# Observation-filtering parameters used for AuRA data collection (they override
# the defaults of aura/perception/phystwin_data_to_observation_pointcloud.py).
VOXEL_SIZE="${VOXEL_SIZE:-0.003}"
STAT_NB_NEIGHBORS="${STAT_NB_NEIGHBORS:-35}"
STAT_STD_RATIO="${STAT_STD_RATIO:-1.2}"
RADIUS_NB_POINTS="${RADIUS_NB_POINTS:-10}"
RADIUS="${RADIUS:-0.006}"
PCA_BODY_LOW="${PCA_BODY_LOW:-1.0}"
PCA_BODY_HIGH="${PCA_BODY_HIGH:-99.0}"
PCA_MARGIN_LONG="${PCA_MARGIN_LONG:-0.0}"
PCA_MARGIN_CROSS="${PCA_MARGIN_CROSS:-0.0}"
TABLE_Z="${TABLE_Z:-0.0}"
TABLE_EPS="${TABLE_EPS:-0.01}"
MASK_OBJECT_ID="${MASK_OBJECT_ID:-0}"
SAM_PROMPT_OVERRIDE="${SAM_PROMPT:-}"

TOP_K="${TOP_K:-20}"
MAX_GRIPPER_WIDTH="${MAX_GRIPPER_WIDTH:-0.08}"
TOP_DOWN="${TOP_DOWN:-1}"
NO_COLLISION_FALLBACK="${NO_COLLISION_FALLBACK:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
ANYGRASP_CHECKPOINT="${ANYGRASP_CHECKPOINT:-}"

RUN_PYROKI="${RUN_PYROKI:-1}"
FILTER_SCRIPT="${FILTER_SCRIPT:-}"
ROBOT_CALIB_RESULTS="${ROBOT_CALIB_RESULTS:-}"
STATE_FILE="${STATE_FILE:-}"
ROBOT_DESCRIPTION="${ROBOT_DESCRIPTION:-fr3_description}"
TARGET_LINK="${TARGET_LINK:-fr3_hand_tcp}"
POSE_FRAME="${POSE_FRAME:-world}"
ROT_OFFSET_RPY=(0 1.57079632679 0)
OFFSET_MODE="${OFFSET_MODE:-post}"
CLEARANCE_M="${CLEARANCE_M:-0.003}"
HOVER_Z_OFFSET="${HOVER_Z_OFFSET:-0.10}"
FINAL_Z_OFFSET="${FINAL_Z_OFFSET:-0.0}"
LIFT_Z_OFFSET="${LIFT_Z_OFFSET:-0.15}"
PATH_SAMPLES="${PATH_SAMPLES:-25}"
HOVER_PATH_LINEAR_STEP_M="${HOVER_PATH_LINEAR_STEP_M:-0.02}"
HOVER_PATH_ANGULAR_STEP_DEG="${HOVER_PATH_ANGULAR_STEP_DEG:-10.0}"
MAX_WAYPOINT_JOINT_DELTA_RAD="${MAX_WAYPOINT_JOINT_DELTA_RAD:-0.25}"
ROBUST_JOINT_MARGIN_RAD="${ROBUST_JOINT_MARGIN_RAD:-0.16}"
MAX_HOME_HOVER_JOINT_DELTA_RAD="${MAX_HOME_HOVER_JOINT_DELTA_RAD:-2.0}"
MAX_HOME_HOVER_TOTAL_JOINT_MOTION_RAD="${MAX_HOME_HOVER_TOTAL_JOINT_MOTION_RAD:-8.0}"
MAX_HOME_HOVER_ROTATION_DEG="${MAX_HOME_HOVER_ROTATION_DEG:-120.0}"
CAMERA_EXTRINSIC_DIR="${CAMERA_EXTRINSIC_DIR:-}"
CAMERA_KEEPOUT_RADIUS_M="${CAMERA_KEEPOUT_RADIUS_M:-0.12}"
MAX_POSITION_ERROR_M="${MAX_POSITION_ERROR_M:-0.001}"
MAX_ORIENTATION_ERROR_DEG="${MAX_ORIENTATION_ERROR_DEG:-0.5}"
USE_SOFT_LIMITS="${USE_SOFT_LIMITS:-1}"

FORCE="${FORCE:-0}"
RESUME="${RESUME:-0}"
REFILTER="${REFILTER:-0}"
CASE_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --force) FORCE=1; shift ;;
    --resume) RESUME=1; shift ;;
    --refilter) REFILTER=1; shift ;;
    --aura-root) need_value "$@"; AURA_REPO_ROOT="$2"; shift 2 ;;
    --data-root) need_value "$@"; DATA_ROOT="$2"; shift 2 ;;
    --aura-env) need_value "$@"; AURA_ENV="$2"; shift 2 ;;
    --anygrasp-env) need_value "$@"; ANYGRASP_ENV="$2"; shift 2 ;;
    --record-resolution) need_value "$@"; RECORD_RESOLUTION="$2"; shift 2 ;;
    --record-fps) need_value "$@"; RECORD_FPS="$2"; shift 2 ;;
    --record-sensor-fps) need_value "$@"; RECORD_SENSOR_FPS="$2"; shift 2 ;;
    --record-sync-fps) need_value "$@"; RECORD_SYNC_FPS="$2"; shift 2 ;;
    --static-sync-tolerance-ms) need_value "$@"; STATIC_SYNC_TOLERANCE_MS="$2"; shift 2 ;;
    --camera-warmup-seconds) need_value "$@"; CAMERA_WARMUP_SECONDS="$2"; shift 2 ;;
    --record-duration) need_value "$@"; RECORD_DURATION="$2"; shift 2 ;;
    --preview) PREVIEW=True; shift ;;
    --no-preview) PREVIEW=False; shift ;;
    --mask-object-id) need_value "$@"; MASK_OBJECT_ID="$2"; shift 2 ;;
    --voxel-size) need_value "$@"; VOXEL_SIZE="$2"; shift 2 ;;
    --stat-nb-neighbors) need_value "$@"; STAT_NB_NEIGHBORS="$2"; shift 2 ;;
    --stat-std-ratio) need_value "$@"; STAT_STD_RATIO="$2"; shift 2 ;;
    --radius-nb-points) need_value "$@"; RADIUS_NB_POINTS="$2"; shift 2 ;;
    --radius) need_value "$@"; RADIUS="$2"; shift 2 ;;
    --pca-body-low) need_value "$@"; PCA_BODY_LOW="$2"; shift 2 ;;
    --pca-body-high) need_value "$@"; PCA_BODY_HIGH="$2"; shift 2 ;;
    --pca-margin-long) need_value "$@"; PCA_MARGIN_LONG="$2"; shift 2 ;;
    --pca-margin-cross) need_value "$@"; PCA_MARGIN_CROSS="$2"; shift 2 ;;
    --table-z) need_value "$@"; TABLE_Z="$2"; shift 2 ;;
    --table-eps) need_value "$@"; TABLE_EPS="$2"; shift 2 ;;
    --sam-prompt) need_value "$@"; SAM_PROMPT_OVERRIDE="$2"; shift 2 ;;
    --top-k) need_value "$@"; TOP_K="$2"; shift 2 ;;
    --max-gripper-width) need_value "$@"; MAX_GRIPPER_WIDTH="$2"; shift 2 ;;
    --top-down) TOP_DOWN=1; shift ;;
    --no-top-down) TOP_DOWN=0; shift ;;
    --collision-fallback) NO_COLLISION_FALLBACK=1; shift ;;
    --no-collision-fallback) NO_COLLISION_FALLBACK=0; shift ;;
    --anygrasp-checkpoint) need_value "$@"; ANYGRASP_CHECKPOINT="$2"; shift 2 ;;
    --omp-num-threads) need_value "$@"; OMP_NUM_THREADS="$2"; shift 2 ;;
    --pyroki-filter) RUN_PYROKI=1; shift ;;
    --no-pyroki-filter) RUN_PYROKI=0; shift ;;
    --filter-script) need_value "$@"; FILTER_SCRIPT="$2"; shift 2 ;;
    --robot-calib-results) need_value "$@"; ROBOT_CALIB_RESULTS="$2"; shift 2 ;;
    --state-file) need_value "$@"; STATE_FILE="$2"; shift 2 ;;
    --description) need_value "$@"; ROBOT_DESCRIPTION="$2"; shift 2 ;;
    --target-link) need_value "$@"; TARGET_LINK="$2"; shift 2 ;;
    --pose-frame) need_value "$@"; POSE_FRAME="$2"; shift 2 ;;
    --rot-offset-rpy)
      [[ $# -ge 4 ]] || die "--rot-offset-rpy requires three values"
      ROT_OFFSET_RPY=("$2" "$3" "$4"); shift 4 ;;
    --offset-mode) need_value "$@"; OFFSET_MODE="$2"; shift 2 ;;
    --clearance-m) need_value "$@"; CLEARANCE_M="$2"; shift 2 ;;
    --hover-z-offset) need_value "$@"; HOVER_Z_OFFSET="$2"; shift 2 ;;
    --final-z-offset) need_value "$@"; FINAL_Z_OFFSET="$2"; shift 2 ;;
    --lift-z-offset) need_value "$@"; LIFT_Z_OFFSET="$2"; shift 2 ;;
    --path-samples) need_value "$@"; PATH_SAMPLES="$2"; shift 2 ;;
    --hover-path-linear-step-m) need_value "$@"; HOVER_PATH_LINEAR_STEP_M="$2"; shift 2 ;;
    --hover-path-angular-step-deg) need_value "$@"; HOVER_PATH_ANGULAR_STEP_DEG="$2"; shift 2 ;;
    --max-waypoint-joint-delta-rad) need_value "$@"; MAX_WAYPOINT_JOINT_DELTA_RAD="$2"; shift 2 ;;
    --robust-joint-margin-rad) need_value "$@"; ROBUST_JOINT_MARGIN_RAD="$2"; shift 2 ;;
    --max-home-hover-joint-delta-rad) need_value "$@"; MAX_HOME_HOVER_JOINT_DELTA_RAD="$2"; shift 2 ;;
    --max-home-hover-total-joint-motion-rad) need_value "$@"; MAX_HOME_HOVER_TOTAL_JOINT_MOTION_RAD="$2"; shift 2 ;;
    --max-home-hover-rotation-deg) need_value "$@"; MAX_HOME_HOVER_ROTATION_DEG="$2"; shift 2 ;;
    --camera-extrinsic-dir) need_value "$@"; CAMERA_EXTRINSIC_DIR="$2"; shift 2 ;;
    --camera-keepout-radius-m) need_value "$@"; CAMERA_KEEPOUT_RADIUS_M="$2"; shift 2 ;;
    --max-position-error-m) need_value "$@"; MAX_POSITION_ERROR_M="$2"; shift 2 ;;
    --max-orientation-error-deg) need_value "$@"; MAX_ORIENTATION_ERROR_DEG="$2"; shift 2 ;;
    --soft-limits) USE_SOFT_LIMITS=1; shift ;;
    --no-soft-limits) USE_SOFT_LIMITS=0; shift ;;
    --*) die "Unknown option: $1" ;;
    *)
      [[ -z "$CASE_NAME" ]] || die "Only one CASE_NAME is allowed"
      CASE_NAME="$1"; shift ;;
  esac
done

[[ -n "$CASE_NAME" ]] || { usage >&2; exit 2; }
[[ "$CASE_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || die "Invalid CASE_NAME: $CASE_NAME"
[[ "$RECORD_RESOLUTION" == "low" || "$RECORD_RESOLUTION" == "high" ]] || die "record resolution must be low or high"
[[ "$POSE_FRAME" == "world" || "$POSE_FRAME" == "base" || "$POSE_FRAME" == "camera" ]] || die "invalid pose frame"
[[ "$OFFSET_MODE" == "post" || "$OFFSET_MODE" == "pre" || "$OFFSET_MODE" == "none" ]] || die "invalid offset mode"
[[ "$FORCE" != "1" || "$RESUME" != "1" ]] || die "--force and --resume are mutually exclusive"

# Release-layout defaults (resolved after --aura-root); config/aura.env or the
# environment override them.
AURA_DATA_ROOT="${AURA_DATA_ROOT:-$AURA_REPO_ROOT/data}"
AURA_CALIB_ROOT="${AURA_CALIB_ROOT:-$AURA_REPO_ROOT/calibration}"
AURA_GSAM_CHECKPOINTS="${AURA_GSAM_CHECKPOINTS:-$AURA_REPO_ROOT/checkpoints/groundedSAM}"
AURA_ANYGRASP_SDK="${AURA_ANYGRASP_SDK:-$AURA_REPO_ROOT/third_party/anygrasp_sdk}"
# The stages change directory (stage 02 runs from the repository, stage 03
# inside the SDK), so a relative AURA_CALIB_ROOT, AURA_ANYGRASP_SDK,
# AURA_GSAM_CHECKPOINTS or AURA_GSAM_REPO is taken from where the script is
# run, like a relative data root; stage 02 expands a leading ~ in the last two.
[[ "$AURA_CALIB_ROOT" == /* ]] || AURA_CALIB_ROOT="$PWD/$AURA_CALIB_ROOT"
[[ "$AURA_ANYGRASP_SDK" == /* ]] || AURA_ANYGRASP_SDK="$PWD/$AURA_ANYGRASP_SDK"
[[ "$AURA_GSAM_CHECKPOINTS" == [/~]* ]] || AURA_GSAM_CHECKPOINTS="$PWD/$AURA_GSAM_CHECKPOINTS"
[[ -z "${AURA_GSAM_REPO:-}" || "$AURA_GSAM_REPO" == [/~]* ]] || AURA_GSAM_REPO="$PWD/$AURA_GSAM_REPO"

# OAK MXIDs in camera-index order, split on any whitespace; checked before capture, with --force before cleanup.
read -r -d '' -a OAK_IDS <<< "${AURA_OAK_IDS:-}" || true
[[ "$FORCE" != "1" || ${#OAK_IDS[@]} -gt 0 ]] || die "AURA_OAK_IDS is empty: set the OAK MXIDs (space separated, camera-index order) in config/aura.env"
[[ -n "$DATA_ROOT" ]] || DATA_ROOT="$AURA_DATA_ROOT/franka_lift"
DATA_ROOT="$(realpath -m -- "$DATA_ROOT")"
[[ "$DATA_ROOT" != "/" ]] || die "Unsafe data root: $DATA_ROOT"
CASE_ROOT="$DATA_ROOT/$CASE_NAME"
ANYGRASP_CASE_DIR="$CASE_ROOT/anygrasp"
PHYSTWIN_DIR="$CASE_ROOT/phystwin"
RESULTS_DIR="$CASE_ROOT/anygrasp_results"
RESULTS_NO_COLLISION_DIR="$CASE_ROOT/anygrasp_results_no_collision"
WORK_ROOT="$CASE_ROOT/_work/anygrasp_collection"
LOG_DIR="$CASE_ROOT/logs"
POINTCLOUD_NPZ="$ANYGRASP_CASE_DIR/observation/observation_frame000_filtered.npz"
POINTCLOUD_PLY="$ANYGRASP_CASE_DIR/observation/observation_frame000_filtered.ply"
SAFE_JSON="$RESULTS_DIR/grasps_table_ik_safe.json"
REPORT_JSON="$RESULTS_DIR/grasps_table_ik_report.json"
SUMMARY_JSON="$CASE_ROOT/one_shot_summary.json"

if [[ -n "$SAM_PROMPT_OVERRIDE" ]]; then
  SAM_PROMPT="$SAM_PROMPT_OVERRIDE"
else
  if [[ "$CASE_NAME" =~ ^(.+)_([0-9]+)$ ]]; then
    PROMPT_STEM="${BASH_REMATCH[1]}"
  else
    PROMPT_STEM="$CASE_NAME"
  fi
  SAM_PROMPT="${PROMPT_STEM//_/ }."
fi

if [[ -e "$CASE_ROOT" && "$FORCE" != "1" && "$RESUME" != "1" ]]; then
  die "Case exists: $CASE_ROOT. Use --resume to skip the stages whose outputs exist, or --force to rebuild preparation outputs from a new capture while preserving phystwin/ (never on a case with a lift recording)."
fi
if [[ "$FORCE" == "1" ]]; then
  echo "[FORCE] Rebuilding preparation outputs while preserving $PHYSTWIN_DIR"
  rm -rf -- "$ANYGRASP_CASE_DIR" "$RESULTS_DIR" "$RESULTS_NO_COLLISION_DIR" "$WORK_ROOT" "$LOG_DIR"
fi

mkdir -p "$CASE_ROOT" "$PHYSTWIN_DIR" "$LOG_DIR" "$WORK_ROOT"

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
else
  CONDA_BASE="$(conda info --base)"
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Some Conda activate.d hooks (e.g. a CUDA setup script in the AnyGrasp env)
# read these variables without guarding against an unset value. This script
# intentionally uses `set -u`, so initialize them before any environment switch.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"

run_stage() {
  local stage_name="$1"
  shift
  echo
  echo "================================================================================"
  echo "[STAGE] $stage_name"
  echo "================================================================================"
  "$@" 2>&1 | tee "$LOG_DIR/${stage_name}.log"
}

cat > "$PHYSTWIN_DIR/README.txt" <<EOF
Reserved for Franka lift recording and PhysTwin-style dynamic data.
Preparation may include offline PyRoki filtering, but this script never commands robot motion.
Case: $CASE_NAME
SAM prompt: $SAM_PROMPT
EOF

echo "[case] root       = $CASE_ROOT"
echo "[case] data root  = $DATA_ROOT"
echo "[case] SAM prompt = $SAM_PROMPT"
echo "[sync] static     = ${STATIC_SYNC_TOLERANCE_MS} ms"

# Stage 1: collect a single static RGB-D frame.
STATIC_READY=0
if [[ -f "$ANYGRASP_CASE_DIR/calibrate.pkl" && -f "$ANYGRASP_CASE_DIR/color/0/0.png" && -f "$ANYGRASP_CASE_DIR/depth/0/0.npy" ]]; then
  STATIC_READY=1
fi
if [[ "$RESUME" == "1" && "$STATIC_READY" == "1" ]]; then
  echo "[SKIP] 01_collect_anygrasp_frame already complete"
else
  [[ ${#OAK_IDS[@]} -gt 0 ]] || die "AURA_OAK_IDS is empty: set the OAK MXIDs (space separated, camera-index order) in config/aura.env"
  conda activate "$AURA_ENV"
  cd "$AURA_REPO_ROOT"
  run_stage "01_collect_anygrasp_frame" \
    python "$AURA_REPO_ROOT/aura/capture/anygrasp_data_collection.py" \
      --project-root "$AURA_REPO_ROOT" \
      --output-root "$ANYGRASP_CASE_DIR" \
      --work-root "$WORK_ROOT" \
      --keep-work-root \
      --overwrite \
      --resolution "$RECORD_RESOLUTION" \
      --fps "$RECORD_FPS" \
      --sensor-fps "$RECORD_SENSOR_FPS" \
      --sync-fps "$RECORD_SYNC_FPS" \
      --warmup-seconds "$CAMERA_WARMUP_SECONDS" \
      --record-duration "$RECORD_DURATION" \
      --capture-bundle-index 0 \
      --sync-tolerance-ms "$STATIC_SYNC_TOLERANCE_MS" \
      --oak-internal-sync-ms "$OAK_INTERNAL_SYNC_MS" \
      --preview "$PREVIEW" \
      --oak-ids "${OAK_IDS[@]}"
fi

# Stage 2: segmentation, fusion, and observation filtering.
if [[ "$RESUME" == "1" && -f "$POINTCLOUD_NPZ" ]]; then
  echo "[SKIP] 02_generate_observation_pointcloud already complete"
else
  conda activate "$AURA_ENV"
  cd "$AURA_REPO_ROOT"
  run_stage "02_generate_observation_pointcloud" \
    python "$AURA_REPO_ROOT/aura/perception/phystwin_data_to_observation_pointcloud.py" \
      "$ANYGRASP_CASE_DIR" \
      --gsam-checkpoints "$AURA_GSAM_CHECKPOINTS" \
      --category "$SAM_PROMPT" \
      --frame 0 \
      --frames-for-pcd 0 \
      --observation-cams 0 1 2 \
      --mask-object-id "$MASK_OBJECT_ID" \
      --voxel-size "$VOXEL_SIZE" \
      --remove-statistical-outlier \
      --stat-nb-neighbors "$STAT_NB_NEIGHBORS" \
      --stat-std-ratio "$STAT_STD_RATIO" \
      --remove-radius-outlier \
      --radius-nb-points "$RADIUS_NB_POINTS" \
      --radius "$RADIUS" \
      --pca-body-filter \
      --pca-body-low "$PCA_BODY_LOW" \
      --pca-body-high "$PCA_BODY_HIGH" \
      --pca-body-margin-long "$PCA_MARGIN_LONG" \
      --pca-body-margin-cross "$PCA_MARGIN_CROSS" \
      --table-filter \
      --z-positive-down \
      --table-z "$TABLE_Z" \
      --table-eps "$TABLE_EPS"
fi
[[ -f "$POINTCLOUD_NPZ" ]] || die "Point cloud missing: $POINTCLOUD_NPZ"

# Stage 3: AnyGrasp inference.
NORMAL_GRASPS="$RESULTS_DIR/grasps.json"
FALLBACK_GRASPS="$RESULTS_NO_COLLISION_DIR/grasps.json"
if [[ "$RESUME" == "1" && ( "$(has_grasps "$NORMAL_GRASPS")" == "1" || "$(has_grasps "$FALLBACK_GRASPS")" == "1" ) ]]; then
  echo "[SKIP] 03_anygrasp_pose_generation already has grasps"
else
  conda activate "$ANYGRASP_ENV"
  export OMP_NUM_THREADS
  SDK="$AURA_ANYGRASP_SDK"
  [[ -d "$SDK/grasp_detection" ]] || die "AnyGrasp SDK not found at $SDK: obtain the SDK and license yourself (see docs/anygrasp_license.md) and set AURA_ANYGRASP_SDK in config/aura.env"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$SDK/grasp_detection:${PYTHONPATH:-}"

  PYTAG="$(python - <<'PY'
import sys
print(f"cpython-{sys.version_info.major}{sys.version_info.minor}")
PY
)"
  if [[ ! -f "$SDK/grasp_detection/gsnet.so" ]]; then
    GSNET_SRC="$(find "$SDK/grasp_detection/gsnet_versions" -name "gsnet.${PYTAG}*.so" -print -quit 2>/dev/null || true)"
    [[ -n "$GSNET_SRC" ]] || die "No gsnet binary for $PYTAG"
    cp "$GSNET_SRC" "$SDK/grasp_detection/gsnet.so"
  fi
  if [[ ! -f "$SDK/grasp_detection/lib_cxx.so" ]]; then
    LIBCXX_SRC="$(find "$SDK/license_registration/lib_cxx_versions" -name "lib_cxx.${PYTAG}*.so" -print -quit 2>/dev/null || true)"
    [[ -n "$LIBCXX_SRC" ]] || die "No lib_cxx binary for $PYTAG"
    cp "$LIBCXX_SRC" "$SDK/grasp_detection/lib_cxx.so"
  fi

  CHECKPOINT_PATH="$ANYGRASP_CHECKPOINT"
  if [[ -z "$CHECKPOINT_PATH" ]]; then
    CHECKPOINT_PATH="$SDK/grasp_detection/checkpoint_detection.tar"
  fi
  if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    CHECKPOINT_PATH="$(find "$SDK" -name checkpoint_detection.tar -print -quit 2>/dev/null || true)"
  fi
  [[ -n "$CHECKPOINT_PATH" && -f "$CHECKPOINT_PATH" ]] || die "Cannot find checkpoint_detection.tar"

  ANYGRASP_ARGS=(
    --input "$POINTCLOUD_NPZ"
    --checkpoint_path "$CHECKPOINT_PATH"
    --output_dir "$RESULTS_DIR"
    --top_k "$TOP_K"
    --max_gripper_width "$MAX_GRIPPER_WIDTH"
    --save_debug_images
  )
  if [[ "$TOP_DOWN" == "1" ]]; then
    ANYGRASP_ARGS+=(--top_down_grasp)
  fi
  cd "$SDK/grasp_detection"
  run_stage "03_anygrasp_pose_generation" \
    python "$AURA_REPO_ROOT/aura/grasping/anygrasp_poses_from_pointcloud.py" "${ANYGRASP_ARGS[@]}"

  if [[ "$(has_grasps "$NORMAL_GRASPS")" != "1" && "$NO_COLLISION_FALLBACK" == "1" ]]; then
    FALLBACK_ARGS=(
      --input "$POINTCLOUD_NPZ"
      --checkpoint_path "$CHECKPOINT_PATH"
      --output_dir "$RESULTS_NO_COLLISION_DIR"
      --top_k "$TOP_K"
      --max_gripper_width "$MAX_GRIPPER_WIDTH"
      --no_collision_detection
      --save_debug_images
    )
    if [[ "$TOP_DOWN" == "1" ]]; then
      FALLBACK_ARGS+=(--top_down_grasp)
    fi
    run_stage "03b_anygrasp_pose_generation_no_collision" \
      python "$AURA_REPO_ROOT/aura/grasping/anygrasp_poses_from_pointcloud.py" "${FALLBACK_ARGS[@]}"
  fi
fi

if [[ "$(has_grasps "$NORMAL_GRASPS")" == "1" ]]; then
  SELECTED_GRASPS="$NORMAL_GRASPS"
elif [[ "$(has_grasps "$FALLBACK_GRASPS")" == "1" ]]; then
  SELECTED_GRASPS="$FALLBACK_GRASPS"
else
  die "AnyGrasp produced no grasps, including fallback"
fi

# Stage 4: offline table/fingertip/PyRoki filtering.
if [[ "$RUN_PYROKI" == "1" ]]; then
  if [[ -z "$FILTER_SCRIPT" ]]; then
    FILTER_SCRIPT="$(first_file \
      "$AURA_REPO_ROOT/aura/grasping/filter_grasps_table_collision_pyroki.py" || true)"
  fi
  [[ -n "$FILTER_SCRIPT" && -f "$FILTER_SCRIPT" ]] || die "Cannot find filter_grasps_table_collision_pyroki.py; pass --filter-script"

  if [[ -z "$ROBOT_CALIB_RESULTS" ]]; then
    ROBOT_CALIB_RESULTS="$AURA_CALIB_ROOT/robot_calibration_results"
  fi
  [[ -e "$ROBOT_CALIB_RESULTS" ]] || die "Robot calibration input not found: $ROBOT_CALIB_RESULTS"

  if [[ -z "$STATE_FILE" ]]; then
    STATE_FILE="$(first_file \
      "$AURA_REPO_ROOT/assets/fr3_home_state.npz" || true)"
  fi
  [[ -n "$STATE_FILE" && -f "$STATE_FILE" ]] || die "Cannot find fr3_home_state.npz; pass --state-file"

  if [[ "$RESUME" == "1" && "$REFILTER" != "1" && -f "$SAFE_JSON" && -f "$REPORT_JSON" ]]; then
    echo "[SKIP] 04_pyroki_filter already complete"
  else
    conda activate "$AURA_ENV"
    cd "$AURA_REPO_ROOT"
    if [[ -z "$CAMERA_EXTRINSIC_DIR" ]]; then
      CAMERA_EXTRINSIC_DIR="$AURA_CALIB_ROOT/extrinsic_results"
    fi
    PYROKI_ARGS=(
      --grasps "$SELECTED_GRASPS"
      --calib-results "$ROBOT_CALIB_RESULTS"
      --state-file "$STATE_FILE"
      --out "$SAFE_JSON"
      --report "$REPORT_JSON"
      --description "$ROBOT_DESCRIPTION"
      --target-link "$TARGET_LINK"
      --pose-frame "$POSE_FRAME"
      --rot-offset-rpy "${ROT_OFFSET_RPY[@]}"
      --offset-mode "$OFFSET_MODE"
      --table-z "$TABLE_Z"
      --clearance-m "$CLEARANCE_M"
      --z-positive-down
      --require-center-above-table
      --hover-z-offset "$HOVER_Z_OFFSET"
      --final-z-offset "$FINAL_Z_OFFSET"
      --lift-z-offset "$LIFT_Z_OFFSET"
      --path-samples "$PATH_SAMPLES"
      --hover-path-linear-step-m "$HOVER_PATH_LINEAR_STEP_M"
      --hover-path-angular-step-deg "$HOVER_PATH_ANGULAR_STEP_DEG"
      --max-waypoint-joint-delta-rad "$MAX_WAYPOINT_JOINT_DELTA_RAD"
      --robust-joint-margin-rad "$ROBUST_JOINT_MARGIN_RAD"
      --max-home-hover-joint-delta-rad "$MAX_HOME_HOVER_JOINT_DELTA_RAD"
      --max-home-hover-total-joint-motion-rad "$MAX_HOME_HOVER_TOTAL_JOINT_MOTION_RAD"
      --max-home-hover-rotation-deg "$MAX_HOME_HOVER_ROTATION_DEG"
      --camera-extrinsic-dir "$CAMERA_EXTRINSIC_DIR"
      --camera-keepout-radius-m "$CAMERA_KEEPOUT_RADIUS_M"
      --max-position-error-m "$MAX_POSITION_ERROR_M"
      --max-orientation-error-deg "$MAX_ORIENTATION_ERROR_DEG"
    )
    if [[ "$USE_SOFT_LIMITS" == "1" ]]; then
      PYROKI_ARGS+=(--use-soft-limits)
    else
      PYROKI_ARGS+=(--disable-soft-limits)
    fi
    run_stage "04_pyroki_filter" python "$FILTER_SCRIPT" "${PYROKI_ARGS[@]}"
  fi
  [[ "$(has_grasps "$SAFE_JSON")" == "1" ]] || die "PyRoki filtering accepted no grasps: $SAFE_JSON"
else
  SAFE_JSON=""
  REPORT_JSON=""
  STATE_FILE=""
fi

# Stage 5: preparation summary.
conda activate "$AURA_ENV"
export CASE_NAME DATA_ROOT SAM_PROMPT CASE_ROOT ANYGRASP_CASE_DIR PHYSTWIN_DIR
export POINTCLOUD_NPZ POINTCLOUD_PLY
export SUMMARY_JSON SELECTED_GRASPS SAFE_JSON REPORT_JSON STATE_FILE ROBOT_CALIB_RESULTS RUN_PYROKI
export STATIC_SYNC_TOLERANCE_MS VOXEL_SIZE STAT_NB_NEIGHBORS STAT_STD_RATIO RADIUS_NB_POINTS RADIUS
export PCA_BODY_LOW PCA_BODY_HIGH PCA_MARGIN_LONG PCA_MARGIN_CROSS TABLE_Z TABLE_EPS
python - <<'PY'
import json, os
from pathlib import Path

def count(path):
    if not path:
        return 0
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return len(value) if isinstance(value, list) else 0
    except Exception:
        return 0

summary = {
    "case_name": os.environ["CASE_NAME"],
    "data_root": os.environ["DATA_ROOT"],
    "sam_prompt": os.environ["SAM_PROMPT"],
    "case_root": os.environ["CASE_ROOT"],
    "anygrasp_dataset": os.environ["ANYGRASP_CASE_DIR"],
    "phystwin_dir": os.environ["PHYSTWIN_DIR"],
    "observation_pointcloud_npz": os.environ["POINTCLOUD_NPZ"],
    "observation_pointcloud_ply": os.environ["POINTCLOUD_PLY"],
    "selected_grasps_input": os.environ["SELECTED_GRASPS"],
    "num_selected_anygrasp_poses": count(os.environ["SELECTED_GRASPS"]),
    "pyroki_filter_enabled": os.environ["RUN_PYROKI"] == "1",
    "safe_grasps": os.environ.get("SAFE_JSON") or None,
    "safe_grasp_count": count(os.environ.get("SAFE_JSON", "")),
    "pyroki_report": os.environ.get("REPORT_JSON") or None,
    "pyroki_state_file": os.environ.get("STATE_FILE") or None,
    "robot_calibration_input": os.environ.get("ROBOT_CALIB_RESULTS") or None,
    "static_sync_tolerance_ms": float(os.environ["STATIC_SYNC_TOLERANCE_MS"]),
    "observation_filter": {
        "voxel_size_m": float(os.environ["VOXEL_SIZE"]),
        "stat_nb_neighbors": int(os.environ["STAT_NB_NEIGHBORS"]),
        "stat_std_ratio": float(os.environ["STAT_STD_RATIO"]),
        "radius_nb_points": int(os.environ["RADIUS_NB_POINTS"]),
        "radius_m": float(os.environ["RADIUS"]),
        "pca_low": float(os.environ["PCA_BODY_LOW"]),
        "pca_high": float(os.environ["PCA_BODY_HIGH"]),
        "pca_margin_long_m": float(os.environ["PCA_MARGIN_LONG"]),
        "pca_margin_cross_m": float(os.environ["PCA_MARGIN_CROSS"]),
        "table_z_m": float(os.environ["TABLE_Z"]),
        "table_eps_m": float(os.environ["TABLE_EPS"]),
        "z_positive_down": True,
        "cameras": [0, 1, 2],
    },
    "robot_motion_commanded": False,
}
Path(os.environ["SUMMARY_JSON"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo
echo "================================================================================"
echo "[DONE] AnyGrasp preparation and PyRoki filtering complete."
echo "Case:        $CASE_ROOT"
echo "Point cloud: $POINTCLOUD_NPZ"
echo "Input poses: $SELECTED_GRASPS"
if [[ "$RUN_PYROKI" == "1" ]]; then
  echo "Safe poses:  $SAFE_JSON"
  echo "Report:      $REPORT_JSON"
fi
echo "Summary:     $SUMMARY_JSON"
echo "No robot motion was commanded."
echo "================================================================================"
