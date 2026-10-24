#!/usr/bin/env bash
set -Eeuo pipefail

# Global AuRA one-shot:
# preparation -> PyRoki filtering -> FR3 grasp/lift -> synchronized RGB-D
# recording -> PhysTwin layout + MP4 previews + preserved calibrations.

_aura_self="${BASH_SOURCE[0]}"; [[ ! -L "$_aura_self" ]] || _aura_self="$(readlink -f -- "$_aura_self")"; AURA_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$_aura_self")/.." && pwd)"
if [[ -f "$AURA_REPO_ROOT/config/aura.env" ]]; then
  source "$AURA_REPO_ROOT/config/aura.env"
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_anygrasp_to_lift_one_shot.sh [options] CASE_NAME [-- PREPARATION_OPTIONS...]

Robot safety and execution:
  --home / --no-home               Default: --home
  --home-after / --no-home-after   Return to Home after successful lift recording; default: enabled
  --release-after / --no-release-after
                                    Open gripper before final Home; default: enabled
  --finalize-existing-recording    Release/Home even when resuming an existing raw recording
  --yes, -y                        Skip all interactive robot checkpoints
  --hover-only                     Known limitation: hangs at hover until Ctrl-C (no release/home); see docs/troubleshooting.md
  --safe-index N                   Index in grasps_table_ik_safe.json; default: 0
  --robot-ip IP                    Default: $AURA_ROBOT_IP (required by every robot stage)
  --motion-speed V                 Average hover-approach speed in m/s; default: 0.10
  --descend-speed V                Average descent speed in m/s; default: 0.02
  --lift-speed V                   Average lift speed in m/s; default: 0.05
  --cartesian-runtime-joint-margin-rad RAD
                                    Abort before a soft joint limit; default: 0.15
  --hover-z-offset M               Default: 0.10
  --final-z-offset M               Default: 0
  --lift-z-offset M                Default: 0.15
  --rot-offset-rpy R P Y           Grasp rotation offset (rad); default: 0 1.57079632679 0
  --motion-start-delta-m M         Default: 0.0015
  --motion-start-timeout-s SEC     Default: 3
  --record-after-lift SEC          Default: 1

Dynamic recording:
  --record-resolution low|high     Default: high
  --record-fps N                   Default: 15
  --record-sensor-fps N            Default: 15
  --record-sync-fps N              Default: 15
  --lift-sync-tolerance-ms MS      Default: 45 (dynamic motion)
  --camera-warmup-seconds SEC      Default: 2
  --state-publish-hz HZ            Default: 100
  --record-max-duration SEC        Default: 60
  --record-preview / --no-record-preview
                                    Default: disabled

Paths and preparation:
  --aura-root PATH                 Repository root for scripts/, aura/ and assets/; default: parent of scripts/
                                    (AURA_* paths already set by config/aura.env or the shell do not move with it)
  --data-root PATH                 Default: $AURA_DATA_ROOT/franka_lift
  --aura-env NAME                  Default: AuRA
  --anygrasp-env NAME              Default: Anygrasp
  --uv-project PATH                Default: $AURA_ROBOT_PROJECT (environments/robot)
  --prep-script PATH               Default: scripts/anygrasp_poses_one_shot.sh
  --execute-script PATH            Default: aura/robot/execute_anygrasp_franka_record_lift.py
  --robot-calib PATH               Default: $AURA_CALIB_ROOT/robot_calibration_results/robot_calib_result.npz
  --state-file PATH                Default: assets/fr3_home_state.npz
  --overwrite                      Delete the entire existing case, including
                                    phystwin/, then rebuild from fresh capture
  --force-preparation              Recapture and rebuild (keeps phystwin/); never on a case with a lift recording
  --resume-preparation             Resume preparation explicitly
  --refilter / --no-refilter       Re-run the filter (earlier filter options are not reused); default: enabled

Arguments after "--" go to anygrasp_poses_one_shot.sh. On a prepared case, options for capture, segmentation
and AnyGrasp need --force-preparation or a new case name; by default only the filter (stage 04) runs again:
  bash scripts/run_anygrasp_to_lift_one_shot.sh green_toy_6 -- --camera-keepout-radius-m 0.15

Configuration (config/aura.env, see config/aura.env.example):
  AURA_ROBOT_IP       Required by every robot stage: FR3 address
  AURA_OAK_IDS        Required: OAK MXIDs, space separated, in camera-index order 0 1 2
  AURA_ROBOT_PROJECT  uv project for the robot stages; default: environments/robot
  AURA_ENV, ANYGRASP_ENV
                      Conda env names; defaults: AuRA, Anygrasp
  AURA_DATA_ROOT      Default: data/ in the repository
  AURA_CALIB_ROOT     Default: calibration/ in the repository; must contain
                      intrinsic_results/1280x720/oak_<MXID>.{npz,json} (plus 640x400/ for
                      --record-resolution low), extrinsic_results/oak_<MXID>.{npz,json},
                      robot_calibration_results/robot_calib_result.npz and
                      robot_calibration_results/fingertip_calibration_results/
                      franka_fingertip_table_calibration_simple.json
Most options also read an environment variable of their own (ROBOT_IP, DATA_ROOT, UV_PROJECT, SAFE_INDEX,
AUTO_YES, ...), which takes precedence over config/aura.env. On/off variables count as on only when set to 1
(RECORD_PREVIEW: True or False). --aura-root, --rot-offset-rpy, --force-preparation and --resume-preparation
are command-line only. POSE_FRAME (default: world) is environment-only.

Run from a shell with the conda env $AURA_ENV (AuRA) active: the
safe-grasp preflight and the final summary run on the python found on PATH.
conda and uv must be on PATH as well.
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

safe_copy() {
  local source="$1"
  local destination="$2"
  [[ -f "$source" ]] || die "Cannot preserve missing file: $source"
  mkdir -p "$(dirname "$destination")"
  if [[ -e "$destination" ]]; then
    cmp -s "$source" "$destination" || die "Refusing to overwrite different preserved file: $destination"
    echo "[reuse] $destination"
  else
    cp --preserve=timestamps "$source" "$destination"
    cmp -s "$source" "$destination" || die "Copied file differs from source: $destination"
    echo "[copy] $destination"
  fi
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

# ------------------------- defaults -------------------------
DATA_ROOT="${DATA_ROOT:-}"
UV_PROJECT="${UV_PROJECT:-}"
AURA_ENV="${AURA_ENV:-AuRA}"
ANYGRASP_ENV="${ANYGRASP_ENV:-Anygrasp}"
PREP_SCRIPT="${PREP_SCRIPT:-}"
EXECUTE_SCRIPT="${EXECUTE_SCRIPT:-}"
ROBOT_CALIB="${ROBOT_CALIB:-}"
STATE_FILE="${STATE_FILE:-}"
STATE_FILE_EXPLICIT=0

HOME_ROBOT="${HOME_ROBOT:-1}"
HOME_AFTER="${HOME_AFTER:-1}"
RELEASE_AFTER="${RELEASE_AFTER:-1}"
FINALIZE_EXISTING_RECORDING="${FINALIZE_EXISTING_RECORDING:-0}"
AUTO_YES="${AUTO_YES:-0}"
HOVER_ONLY="${HOVER_ONLY:-0}"
SAFE_INDEX="${SAFE_INDEX:-0}"
ROBOT_IP="${ROBOT_IP:-${AURA_ROBOT_IP:-}}"
MOTION_SPEED="${MOTION_SPEED:-0.10}"
DESCEND_SPEED="${DESCEND_SPEED:-0.02}"
LIFT_SPEED="${LIFT_SPEED:-0.05}"
CARTESIAN_RUNTIME_JOINT_MARGIN_RAD="${CARTESIAN_RUNTIME_JOINT_MARGIN_RAD:-0.15}"
HOVER_Z_OFFSET="${HOVER_Z_OFFSET:-0.10}"
FINAL_Z_OFFSET="${FINAL_Z_OFFSET:-0.0}"
LIFT_Z_OFFSET="${LIFT_Z_OFFSET:-0.15}"
ROT_OFFSET_RPY=(0 1.57079632679 0)
POSE_FRAME="${POSE_FRAME:-world}"
MOTION_START_DELTA_M="${MOTION_START_DELTA_M:-0.0015}"
MOTION_START_TIMEOUT_S="${MOTION_START_TIMEOUT_S:-3.0}"
RECORD_AFTER_LIFT="${RECORD_AFTER_LIFT:-1.0}"

RECORD_RESOLUTION="${RECORD_RESOLUTION:-high}"
RECORD_FPS="${RECORD_FPS:-15}"
RECORD_SENSOR_FPS="${RECORD_SENSOR_FPS:-15}"
RECORD_SYNC_FPS="${RECORD_SYNC_FPS:-15}"
LIFT_SYNC_TOLERANCE_MS="${LIFT_SYNC_TOLERANCE_MS:-45}"
CAMERA_WARMUP_SECONDS="${CAMERA_WARMUP_SECONDS:-2.0}"
STATE_PUBLISH_HZ="${STATE_PUBLISH_HZ:-100}"
RECORD_MAX_DURATION="${RECORD_MAX_DURATION:-60}"
RECORD_PREVIEW="${RECORD_PREVIEW:-False}"

PREP_MODE="auto"
OVERWRITE="${OVERWRITE:-0}"
REFILTER="${REFILTER:-1}"
CASE_NAME=""
PREP_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --)
      shift
      PREP_EXTRA_ARGS=("$@")
      break ;;
    --home) HOME_ROBOT=1; shift ;;
    --no-home) HOME_ROBOT=0; shift ;;
    --home-after) HOME_AFTER=1; shift ;;
    --no-home-after) HOME_AFTER=0; shift ;;
    --release-after) RELEASE_AFTER=1; shift ;;
    --no-release-after) RELEASE_AFTER=0; shift ;;
    --finalize-existing-recording) FINALIZE_EXISTING_RECORDING=1; shift ;;
    --yes|-y) AUTO_YES=1; shift ;;
    --hover-only) HOVER_ONLY=1; shift ;;
    --safe-index) need_value "$@"; SAFE_INDEX="$2"; shift 2 ;;
    --robot-ip) need_value "$@"; ROBOT_IP="$2"; shift 2 ;;
    --motion-speed) need_value "$@"; MOTION_SPEED="$2"; shift 2 ;;
    --descend-speed) need_value "$@"; DESCEND_SPEED="$2"; shift 2 ;;
    --lift-speed) need_value "$@"; LIFT_SPEED="$2"; shift 2 ;;
    --cartesian-runtime-joint-margin-rad) need_value "$@"; CARTESIAN_RUNTIME_JOINT_MARGIN_RAD="$2"; shift 2 ;;
    --hover-z-offset) need_value "$@"; HOVER_Z_OFFSET="$2"; shift 2 ;;
    --final-z-offset) need_value "$@"; FINAL_Z_OFFSET="$2"; shift 2 ;;
    --lift-z-offset) need_value "$@"; LIFT_Z_OFFSET="$2"; shift 2 ;;
    --rot-offset-rpy)
      [[ $# -ge 4 ]] || die "--rot-offset-rpy requires three values"
      ROT_OFFSET_RPY=("$2" "$3" "$4"); shift 4 ;;
    --motion-start-delta-m) need_value "$@"; MOTION_START_DELTA_M="$2"; shift 2 ;;
    --motion-start-timeout-s) need_value "$@"; MOTION_START_TIMEOUT_S="$2"; shift 2 ;;
    --record-after-lift) need_value "$@"; RECORD_AFTER_LIFT="$2"; shift 2 ;;
    --record-resolution) need_value "$@"; RECORD_RESOLUTION="$2"; shift 2 ;;
    --record-fps) need_value "$@"; RECORD_FPS="$2"; shift 2 ;;
    --record-sensor-fps) need_value "$@"; RECORD_SENSOR_FPS="$2"; shift 2 ;;
    --record-sync-fps) need_value "$@"; RECORD_SYNC_FPS="$2"; shift 2 ;;
    --lift-sync-tolerance-ms) need_value "$@"; LIFT_SYNC_TOLERANCE_MS="$2"; shift 2 ;;
    --camera-warmup-seconds) need_value "$@"; CAMERA_WARMUP_SECONDS="$2"; shift 2 ;;
    --state-publish-hz) need_value "$@"; STATE_PUBLISH_HZ="$2"; shift 2 ;;
    --record-max-duration) need_value "$@"; RECORD_MAX_DURATION="$2"; shift 2 ;;
    --record-preview) RECORD_PREVIEW=True; shift ;;
    --no-record-preview) RECORD_PREVIEW=False; shift ;;
    --aura-root) need_value "$@"; AURA_REPO_ROOT="$2"; shift 2 ;;
    --data-root) need_value "$@"; DATA_ROOT="$2"; shift 2 ;;
    --aura-env) need_value "$@"; AURA_ENV="$2"; shift 2 ;;
    --anygrasp-env) need_value "$@"; ANYGRASP_ENV="$2"; shift 2 ;;
    --uv-project) need_value "$@"; UV_PROJECT="$2"; shift 2 ;;
    --prep-script) need_value "$@"; PREP_SCRIPT="$2"; shift 2 ;;
    --execute-script) need_value "$@"; EXECUTE_SCRIPT="$2"; shift 2 ;;
    --robot-calib) need_value "$@"; ROBOT_CALIB="$2"; shift 2 ;;
    --state-file) need_value "$@"; STATE_FILE="$2"; STATE_FILE_EXPLICIT=1; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --force-preparation) PREP_MODE=force; shift ;;
    --resume-preparation) PREP_MODE=resume; shift ;;
    --refilter) REFILTER=1; shift ;;
    --no-refilter) REFILTER=0; shift ;;
    --*) die "Unknown global option: $1. Put preparation-only options after --" ;;
    *)
      [[ -z "$CASE_NAME" ]] || die "Only one CASE_NAME is allowed"
      CASE_NAME="$1"; shift ;;
  esac
done

[[ -n "$CASE_NAME" ]] || { usage >&2; exit 2; }
[[ "$CASE_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || die "Invalid CASE_NAME: $CASE_NAME"
[[ "$RECORD_RESOLUTION" == "low" || "$RECORD_RESOLUTION" == "high" ]] || die "record resolution must be low or high"
[[ "$SAFE_INDEX" =~ ^[0-9]+$ ]] || die "safe-index must be a non-negative integer"
[[ "$OVERWRITE" == "0" || "$OVERWRITE" == "1" ]] || die "OVERWRITE must be 0 or 1"
[[ "$OVERWRITE" != "1" || "$PREP_MODE" == "auto" ]] || die "--overwrite cannot be combined with --force-preparation or --resume-preparation"
AURA_DATA_ROOT="${AURA_DATA_ROOT:-$AURA_REPO_ROOT/data}"
AURA_CALIB_ROOT="${AURA_CALIB_ROOT:-$AURA_REPO_ROOT/calibration}"
# The preparation script changes directory (stage 03 runs inside the SDK) and receives
# --robot-calib-results built from this root, so a relative AURA_CALIB_ROOT is taken
# from where this script is run, as in anygrasp_poses_one_shot.sh.
[[ "$AURA_CALIB_ROOT" == /* ]] || AURA_CALIB_ROOT="$PWD/$AURA_CALIB_ROOT"
[[ -n "$UV_PROJECT" ]] || UV_PROJECT="${AURA_ROBOT_PROJECT:-$AURA_REPO_ROOT/environments/robot}"
[[ -d "$UV_PROJECT" ]] || die "uv project not found: $UV_PROJECT"
command -v uv >/dev/null 2>&1 || die "uv command not found"

[[ -n "$DATA_ROOT" ]] || DATA_ROOT="$AURA_DATA_ROOT/franka_lift"
DATA_ROOT="$(realpath -m -- "$DATA_ROOT")"
[[ "$DATA_ROOT" != "/" ]] || die "Unsafe data root: $DATA_ROOT"
CASE_ROOT="$DATA_ROOT/$CASE_NAME"
PHYSTWIN_DIR="$CASE_ROOT/phystwin"
RAW_RECORDING_DIR="$PHYSTWIN_DIR/raw_recording"
SAFE_JSON="$CASE_ROOT/anygrasp_results/grasps_table_ik_safe.json"
GLOBAL_SUMMARY="$CASE_ROOT/global_one_shot_summary.json"

# The robot IP is needed only if stage 02 or 02b will connect. Stage 01 never
# touches phystwin/raw_recording, so its state here decides (unless --overwrite
# deletes it): an incomplete recording (only one of bundle_* and
# recording_summary.json) stops the run after stage 01, before any robot stage;
# a complete one skips stage 02, and 02b then runs only for
# --finalize-existing-recording with release or home enabled.
ROBOT_NEEDED=1
if [[ "$OVERWRITE" != "1" ]]; then
  PRE_RAW_BUNDLES=0
  PRE_RAW_SUMMARY=0
  if compgen -G "$RAW_RECORDING_DIR/bundle_*" >/dev/null; then
    PRE_RAW_BUNDLES=1
  fi
  if [[ -f "$RAW_RECORDING_DIR/recording_summary.json" ]]; then
    PRE_RAW_SUMMARY=1
  fi
  if [[ "$PRE_RAW_BUNDLES" == "1" || "$PRE_RAW_SUMMARY" == "1" ]]; then
    ROBOT_NEEDED=0
    if [[ "$PRE_RAW_BUNDLES" == "1" && "$PRE_RAW_SUMMARY" == "1" && "$FINALIZE_EXISTING_RECORDING" == "1" \
      && ( "$RELEASE_AFTER" == "1" || "$HOME_AFTER" == "1" ) ]]; then
      ROBOT_NEEDED=1
    fi
  fi
fi
[[ "$ROBOT_NEEDED" == "0" || -n "$ROBOT_IP" ]] || die "Robot IP not set: set AURA_ROBOT_IP in config/aura.env or pass --robot-ip"

# Camera-index order 0, 1, 2, split on any whitespace; recordings and calibration files are named oak_<MXID>.
read -r -d '' -a OAK_IDS <<< "${AURA_OAK_IDS:-}" || true
[[ "${#OAK_IDS[@]}" -gt 0 ]] || die "AURA_OAK_IDS is empty: set the OAK MXIDs (space separated, camera-index order) in config/aura.env"
CAMERA_ORDER=("${OAK_IDS[@]/#/oak_}")

if [[ -z "$PREP_SCRIPT" ]]; then
  PREP_SCRIPT="$(first_file \
    "$AURA_REPO_ROOT/scripts/anygrasp_poses_one_shot.sh" \
    "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/anygrasp_poses_one_shot.sh" || true)"
fi
[[ -n "$PREP_SCRIPT" && -f "$PREP_SCRIPT" ]] || die "Cannot find anygrasp_poses_one_shot.sh; pass --prep-script"

if [[ -z "$EXECUTE_SCRIPT" ]]; then
  EXECUTE_SCRIPT="$AURA_REPO_ROOT/aura/robot/execute_anygrasp_franka_record_lift.py"
fi
[[ -n "$EXECUTE_SCRIPT" && -f "$EXECUTE_SCRIPT" ]] || die "Cannot find execute_anygrasp_franka_record_lift.py"

if [[ -z "$ROBOT_CALIB" ]]; then
  ROBOT_CALIB="$AURA_CALIB_ROOT/robot_calibration_results/robot_calib_result.npz"
fi
[[ -f "$ROBOT_CALIB" ]] || die "Robot calibration not found: $ROBOT_CALIB"
ROBOT_CALIB_RESULTS="$(dirname "$ROBOT_CALIB")"

if [[ -z "$STATE_FILE" ]]; then
  STATE_FILE="$AURA_REPO_ROOT/assets/fr3_home_state.npz"
fi
[[ -n "$STATE_FILE" && -f "$STATE_FILE" ]] || die "Cannot find fr3_home_state.npz; pass --state-file"
if [[ "$HOME_ROBOT" != "1" && "$STATE_FILE_EXPLICIT" != "1" ]]; then
  die "--no-home requires an explicit fresh --state-file so PyRoki starts from the same robot configuration"
fi

RAW_CONVERT_SCRIPT="$AURA_REPO_ROOT/aura/conversion/raw_data_to_phystwin_style.py"
CALIB_CONVERT_SCRIPT="$AURA_REPO_ROOT/aura/conversion/calibration_matrix_to_phystwin_style.py"
[[ -f "$RAW_CONVERT_SCRIPT" ]] || die "Missing converter: $RAW_CONVERT_SCRIPT"
[[ -f "$CALIB_CONVERT_SCRIPT" ]] || die "Missing converter: $CALIB_CONVERT_SCRIPT"

if [[ "$RECORD_RESOLUTION" == "high" ]]; then
  INTRINSIC_DIR="$AURA_CALIB_ROOT/intrinsic_results/1280x720"
else
  INTRINSIC_DIR="$AURA_CALIB_ROOT/intrinsic_results/640x400"
fi
EXTRINSIC_DIR="$AURA_CALIB_ROOT/extrinsic_results"
[[ -d "$INTRINSIC_DIR" ]] || die "Intrinsic directory missing: $INTRINSIC_DIR"
[[ -d "$EXTRINSIC_DIR" ]] || die "Extrinsic directory missing: $EXTRINSIC_DIR"

# A true full-case overwrite is intentionally separate from --force-preparation:
# it also removes stale/partial dynamic recordings and derived PhysTwin data.
# Resolve and verify the exact target before deleting anything.
if [[ "$OVERWRITE" == "1" ]]; then
  DATA_ROOT_CANONICAL="$(realpath -m -- "$DATA_ROOT")"
  CASE_ROOT_CANONICAL="$(realpath -m -- "$CASE_ROOT")"
  EXPECTED_CASE_ROOT="$DATA_ROOT_CANONICAL/$CASE_NAME"
  [[ "$CASE_ROOT_CANONICAL" == "$EXPECTED_CASE_ROOT" ]] || \
    die "Refusing overwrite: resolved case path is unexpected: $CASE_ROOT_CANONICAL"
  [[ "$CASE_ROOT_CANONICAL" != "/" && "$CASE_ROOT_CANONICAL" != "$DATA_ROOT_CANONICAL" ]] || \
    die "Refusing overwrite of unsafe path: $CASE_ROOT_CANONICAL"

  if [[ -e "$CASE_ROOT_CANONICAL" ]]; then
    echo "[OVERWRITE] Deleting entire existing case: $CASE_ROOT_CANONICAL"
    echo "[OVERWRITE] This includes anygrasp outputs, logs, raw recordings, and phystwin data."
    rm -rf -- "$CASE_ROOT_CANONICAL"
    [[ ! -e "$CASE_ROOT_CANONICAL" ]] || die "Failed to remove existing case: $CASE_ROOT_CANONICAL"
  else
    echo "[OVERWRITE] No existing case found; starting fresh: $CASE_ROOT_CANONICAL"
  fi
fi

# Preparation defaults to auto-resume when a case already exists.
PREP_MODE_ARGS=()
if [[ "$PREP_MODE" == "force" ]]; then
  PREP_MODE_ARGS+=(--force)
elif [[ "$PREP_MODE" == "resume" || -e "$CASE_ROOT" ]]; then
  PREP_MODE_ARGS+=(--resume)
fi

PREP_ARGS=(
  "${PREP_MODE_ARGS[@]}"
  --aura-root "$AURA_REPO_ROOT"
  --data-root "$DATA_ROOT"
  --aura-env "$AURA_ENV"
  --anygrasp-env "$ANYGRASP_ENV"
  --robot-calib-results "$ROBOT_CALIB_RESULTS"
  --state-file "$STATE_FILE"
  --pose-frame "$POSE_FRAME"
  --rot-offset-rpy "${ROT_OFFSET_RPY[@]}"
  --hover-z-offset "$HOVER_Z_OFFSET"
  --final-z-offset "$FINAL_Z_OFFSET"
  --lift-z-offset "$LIFT_Z_OFFSET"
  --pyroki-filter
)
if [[ "$REFILTER" == "1" ]]; then
  PREP_ARGS+=(--refilter)
fi
PREP_ARGS+=("${PREP_EXTRA_ARGS[@]}")
PREP_ARGS+=("$CASE_NAME")

run_stage "01_prepare_and_filter" bash "$PREP_SCRIPT" "${PREP_ARGS[@]}"

[[ -f "$SAFE_JSON" ]] || die "Safe grasp list missing: $SAFE_JSON"
# Stdlib-only helper (like the final summary): runs on the python on PATH, i.e. the active AuRA env.
python - "$SAFE_JSON" "$SAFE_INDEX" <<'PY'
import json, sys
p, index = sys.argv[1], int(sys.argv[2])
value = json.load(open(p, "r", encoding="utf-8"))
if not isinstance(value, list) or not value:
    raise SystemExit(f"No safe grasps in {p}")
if index >= len(value):
    raise SystemExit(f"safe-index {index} is out of range; accepted grasps: {len(value)}")
g = value[index]
print("=" * 72)
print(f"Selected safe-list index: {index}")
print(f"Original AnyGrasp rank:   {g.get('rank', 'unknown')}")
print(f"AnyGrasp score:           {g.get('score', 'unknown')}")
check = g.get("table_ik_fk_check", {})
tip_check = check.get("target_fingertip_table_check", {})
clearance = tip_check.get("minimum_clearance_m")
print(f"Target tip clearance:     {clearance if clearance is not None else 'see report'}")
robust = check.get("home_to_hover_cartesian_robustness", {})
if not robust.get("safe", False):
    raise SystemExit("Selected grasp has no passing dense Cartesian robustness report; run --refilter")
print(f"Cartesian rotation:       {robust.get('cartesian_rotation_deg')} deg")
print(f"Max endpoint joint delta: {robust.get('endpoint_max_joint_delta_rad')} rad")
print(f"Minimum soft margin:      {robust.get('minimum_soft_joint_margin_rad')} rad")
print(f"Minimum camera distance:  {robust.get('minimum_camera_distance_m')} m")
print("=" * 72)
PY

REUSE_RAW_RECORDING=0
HAS_RAW_BUNDLES=0
HAS_RAW_SUMMARY=0
if compgen -G "$RAW_RECORDING_DIR/bundle_*" >/dev/null; then
  HAS_RAW_BUNDLES=1
fi
if [[ -f "$RAW_RECORDING_DIR/recording_summary.json" ]]; then
  HAS_RAW_SUMMARY=1
fi
if [[ "$HAS_RAW_BUNDLES" == "1" && "$HAS_RAW_SUMMARY" == "1" ]]; then
  REUSE_RAW_RECORDING=1
  echo "[resume] Complete raw lift recording already exists; robot execution will be skipped."
  echo "[resume] Raw recording: $RAW_RECORDING_DIR"
elif [[ "$HAS_RAW_BUNDLES" == "1" || "$HAS_RAW_SUMMARY" == "1" ]]; then
  die "Raw recording is incomplete (bundles=$HAS_RAW_BUNDLES summary=$HAS_RAW_SUMMARY): $RAW_RECORDING_DIR"
fi

EXEC_ARGS=(
  --ip "$ROBOT_IP"
  --calib "$ROBOT_CALIB"
  --grasp-json "$SAFE_JSON"
  --rank "$SAFE_INDEX"
  --pose-frame "$POSE_FRAME"
  --orientation-mode anygrasp
  --rot-offset-rpy "${ROT_OFFSET_RPY[@]}"
  --execute
  --record-output-root "$RAW_RECORDING_DIR"
  --record-oak-ids "${OAK_IDS[@]}"
  --record-resolution "$RECORD_RESOLUTION"
  --record-fps "$RECORD_FPS"
  --record-sensor-fps "$RECORD_SENSOR_FPS"
  --record-sync-fps "$RECORD_SYNC_FPS"
  --record-sync-tolerance-ms "$LIFT_SYNC_TOLERANCE_MS"
  --camera-warmup-seconds "$CAMERA_WARMUP_SECONDS"
  --state-publish-hz "$STATE_PUBLISH_HZ"
  --record-max-duration "$RECORD_MAX_DURATION"
  --record-preview "$RECORD_PREVIEW"
  --motion-speed "$MOTION_SPEED"
  --descend-speed "$DESCEND_SPEED"
  --lift-speed "$LIFT_SPEED"
  --cartesian-runtime-joint-margin-rad "$CARTESIAN_RUNTIME_JOINT_MARGIN_RAD"
  --hover-z-offset "$HOVER_Z_OFFSET"
  --final-z-offset "$FINAL_Z_OFFSET"
  --lift-z-offset "$LIFT_Z_OFFSET"
  --motion-start-delta-m "$MOTION_START_DELTA_M"
  --motion-start-timeout-s "$MOTION_START_TIMEOUT_S"
  --record-after-lift "$RECORD_AFTER_LIFT"
)
if [[ "$HOME_ROBOT" == "1" ]]; then
  EXEC_ARGS+=(--home)
fi
if [[ "$AUTO_YES" == "1" ]]; then
  EXEC_ARGS+=(--yes)
fi
if [[ "$HOVER_ONLY" == "1" ]]; then
  EXEC_ARGS+=(--hover-only)
fi

ROBOT_EXECUTED=0
if [[ "$REUSE_RAW_RECORDING" == "1" ]]; then
  echo "[SKIP] 02_franka_grasp_lift_record already complete"
else
  run_stage "02_franka_grasp_lift_record" \
    uv run --project "$UV_PROJECT" python "$EXECUTE_SCRIPT" "${EXEC_ARGS[@]}"
  ROBOT_EXECUTED=1
fi

FINALIZE_ROBOT=0
if [[ "$ROBOT_EXECUTED" == "1" ]]; then
  FINALIZE_ROBOT=1
elif [[ "$REUSE_RAW_RECORDING" == "1" && "$FINALIZE_EXISTING_RECORDING" == "1" ]]; then
  FINALIZE_ROBOT=1
fi

if [[ "$FINALIZE_ROBOT" == "1" && ( "$RELEASE_AFTER" == "1" || "$HOME_AFTER" == "1" ) ]]; then
  FINAL_ACTIONS=""
  if [[ "$RELEASE_AFTER" == "1" ]]; then
    FINAL_ACTIONS="release the grasp at the current pose"
  fi
  if [[ "$HOME_AFTER" == "1" ]]; then
    if [[ -n "$FINAL_ACTIONS" ]]; then
      FINAL_ACTIONS="$FINAL_ACTIONS, then return HOME"
    else
      FINAL_ACTIONS="return HOME"
    fi
  fi
  if [[ "$AUTO_YES" == "1" ]]; then
    echo "[checkpoint][auto-yes] Final robot action: $FINAL_ACTIONS"
  else
    echo
    echo "============================================================"
    echo "ROBOT CHECKPOINT"
    echo "Final robot action: $FINAL_ACTIONS"
    if [[ "$RELEASE_AFTER" == "1" ]]; then
      echo "WARNING: the held object will be released at the robot's current pose."
    fi
    echo "Press Enter to continue (y/yes also works); anything else aborts."
    read -r -p "> " FINAL_ACTION_REPLY
    case "${FINAL_ACTION_REPLY,,}" in
      ""|y|yes) ;;
      *) die "Final robot action aborted by user" ;;
    esac
  fi

  echo
  echo "================================================================================"
  echo "[STAGE] 02b_release_and_return_home"
  echo "================================================================================"
  uv run --project "$UV_PROJECT" python - "$ROBOT_IP" "$RELEASE_AFTER" "$HOME_AFTER" <<'PY'
import sys
import fr3

ip = sys.argv[1]
release_after = sys.argv[2] == "1"
home_after = sys.argv[3] == "1"
fr3.ROBOT_IP = ip
print(f"[robot] connecting to {ip} for final actions")
with fr3.FR3(ip=ip) as robot:
    if release_after:
        print("[robot] opening gripper to release grasp")
        robot.open_gripper()
        print("[robot] grasp released")
    if home_after:
        print("[robot] returning Home")
        robot.home(blocking=True)
        print("[robot] final Home complete")
PY
fi

if [[ "$HOVER_ONLY" == "1" ]]; then
  echo "[DONE] Hover-only completed. No dynamic recording conversion was attempted."
  exit 0
fi

compgen -G "$RAW_RECORDING_DIR/bundle_*" >/dev/null || die "Execution finished without raw recording bundles"

# Use conda run so AuRA conversion dependencies do not leak into the uv process.
run_stage "03_convert_dynamic_to_phystwin" \
  conda run --no-capture-output -n "$AURA_ENV" \
    python "$RAW_CONVERT_SCRIPT" \
      --raw-root "$RAW_RECORDING_DIR" \
      --output-root "$PHYSTWIN_DIR" \
      --camera-order "${CAMERA_ORDER[@]}" \
      --start-frame 0 \
      --stride 1 \
      --copy-meta \
      --overwrite

# Preserve conversion metadata before camera calibration writes metadata.json.
[[ -f "$PHYSTWIN_DIR/metadata.json" ]] || die "Dynamic conversion did not create metadata.json"
cp --preserve=timestamps "$PHYSTWIN_DIR/metadata.json" "$PHYSTWIN_DIR/conversion_metadata.json"

FRAME_COUNT="$(find "$PHYSTWIN_DIR/color/0" -maxdepth 1 -type f -name '*.png' | wc -l)"
[[ "$FRAME_COUNT" -gt 0 ]] || die "No converted RGB frames found"

# The AuRA OpenCV build may not provide an H.264/avc1 encoder. Generate MP4
# previews separately with MPEG-4 Part 2 (mp4v), which is broadly available.
echo
echo "================================================================================"
echo "[STAGE] 03b_generate_mp4_previews"
echo "================================================================================"
conda run --no-capture-output -n "$AURA_ENV" \
  python - "$PHYSTWIN_DIR" "$RECORD_FPS" <<'PY'
import sys
from pathlib import Path

import cv2

root = Path(sys.argv[1])
fps = float(sys.argv[2])
for camera_index in range(3):
    frame_dir = root / "color" / str(camera_index)
    frames = sorted(frame_dir.glob("*.png"), key=lambda p: int(p.stem))
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
        temporary.unlink(missing_ok=True)
        candidate = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if candidate.isOpened():
            writer = candidate
            selected_codec = codec
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"OpenCV cannot open an MP4 writer using mp4v/FMP4/MJPG: {temporary}")
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

run_stage "04_generate_dynamic_camera_calibration" \
  conda run --no-capture-output -n "$AURA_ENV" \
    python "$CALIB_CONVERT_SCRIPT" \
      --intrinsic-dir "$INTRINSIC_DIR" \
      --extrinsic-dir "$EXTRINSIC_DIR" \
      --output-dir "$PHYSTWIN_DIR" \
      --camera-order "${CAMERA_ORDER[@]}" \
      --serial-numbers "${CAMERA_ORDER[@]}" \
      --frame-num "$FRAME_COUNT" \
      --force-single-wh \
      --overwrite

# Preserve the exact source calibration/state files under fixed, readable names.
safe_copy "$ROBOT_CALIB" "$PHYSTWIN_DIR/robot_calib_result.npz"
safe_copy "$STATE_FILE" "$PHYSTWIN_DIR/fr3_home_state.npz"

FINGERTIP_CALIB="$(first_file \
  "$ROBOT_CALIB_RESULTS/fingertip_calibration_results/franka_fingertip_table_calibration_simple.json" \
  "$ROBOT_CALIB_RESULTS/franka_fingertip_table_calibration_simple.json" || true)"
[[ -n "$FINGERTIP_CALIB" ]] || die "Fingertip calibration not found under $ROBOT_CALIB_RESULTS"
safe_copy "$FINGERTIP_CALIB" "$PHYSTWIN_DIR/franka_fingertip_table_calibration_simple.json"

for camera in "${CAMERA_ORDER[@]}"; do
  safe_copy "$INTRINSIC_DIR/$camera.npz" "$PHYSTWIN_DIR/camera_calibration/intrinsic/$camera.npz"
  safe_copy "$EXTRINSIC_DIR/$camera.npz" "$PHYSTWIN_DIR/camera_calibration/extrinsic/$camera.npz"
  if [[ -f "$INTRINSIC_DIR/$camera.json" ]]; then
    safe_copy "$INTRINSIC_DIR/$camera.json" "$PHYSTWIN_DIR/camera_calibration/intrinsic/$camera.json"
  fi
  if [[ -f "$EXTRINSIC_DIR/$camera.json" ]]; then
    safe_copy "$EXTRINSIC_DIR/$camera.json" "$PHYSTWIN_DIR/camera_calibration/extrinsic/$camera.json"
  fi
done

for summary_name in recording_summary.json session_summary.json; do
  if [[ -f "$RAW_RECORDING_DIR/$summary_name" ]]; then
    cp --preserve=timestamps "$RAW_RECORDING_DIR/$summary_name" "$PHYSTWIN_DIR/$summary_name"
  fi
done

export CASE_NAME DATA_ROOT CASE_ROOT PHYSTWIN_DIR RAW_RECORDING_DIR SAFE_JSON SAFE_INDEX FRAME_COUNT
export HOME_ROBOT HOME_AFTER RELEASE_AFTER AUTO_YES RECORD_RESOLUTION RECORD_FPS RECORD_SENSOR_FPS RECORD_SYNC_FPS
export LIFT_SYNC_TOLERANCE_MS MOTION_SPEED DESCEND_SPEED LIFT_SPEED MOTION_START_DELTA_M RECORD_AFTER_LIFT
export CARTESIAN_RUNTIME_JOINT_MARGIN_RAD REFILTER
export GLOBAL_SUMMARY
python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["PHYSTWIN_DIR"])
summary = {
    "case_name": os.environ["CASE_NAME"],
    "data_root": os.environ["DATA_ROOT"],
    "case_root": os.environ["CASE_ROOT"],
    "safe_grasps": os.environ["SAFE_JSON"],
    "safe_index": int(os.environ["SAFE_INDEX"]),
    "home_enabled": os.environ["HOME_ROBOT"] == "1",
    "home_after_enabled": os.environ["HOME_AFTER"] == "1",
    "release_after_enabled": os.environ["RELEASE_AFTER"] == "1",
    "auto_confirmed": os.environ["AUTO_YES"] == "1",
    "dynamic_recording": {
        "raw_bundles": os.environ["RAW_RECORDING_DIR"],
        "phys_twin_root": str(root),
        "num_frames": int(os.environ["FRAME_COUNT"]),
        "resolution": os.environ["RECORD_RESOLUTION"],
        "record_fps": int(os.environ["RECORD_FPS"]),
        "sensor_fps": int(os.environ["RECORD_SENSOR_FPS"]),
        "sync_fps": float(os.environ["RECORD_SYNC_FPS"]),
        "sync_tolerance_ms": float(os.environ["LIFT_SYNC_TOLERANCE_MS"]),
        "hover_speed": float(os.environ["MOTION_SPEED"]),
        "descend_speed": float(os.environ["DESCEND_SPEED"]),
        "lift_speed": float(os.environ["LIFT_SPEED"]),
        "home_to_hover_execution": "direct_cartesian_pose_fci_after_dense_pyroki_robustness_filter",
        "cartesian_runtime_joint_margin_rad": float(os.environ["CARTESIAN_RUNTIME_JOINT_MARGIN_RAD"]),
        "refilter_enabled": os.environ["REFILTER"] == "1",
        "motion_start_delta_m": float(os.environ["MOTION_START_DELTA_M"]),
        "record_after_lift_s": float(os.environ["RECORD_AFTER_LIFT"]),
    },
    "outputs": {
        "color": str(root / "color"),
        "depth": str(root / "depth"),
        "source_meta": str(root / "source_meta"),
        "camera_videos": [str(root / "color" / f"{i}.mp4") for i in range(3)],
        "calibrate_pkl": str(root / "calibrate.pkl"),
        "metadata": str(root / "metadata.json"),
        "conversion_metadata": str(root / "conversion_metadata.json"),
        "robot_calibration": str(root / "robot_calib_result.npz"),
        "fingertip_calibration": str(root / "franka_fingertip_table_calibration_simple.json"),
        "start_state": str(root / "fr3_home_state.npz"),
        "camera_calibration_sources": str(root / "camera_calibration"),
    },
}
required = [
    root / "calibrate.pkl",
    root / "metadata.json",
    root / "conversion_metadata.json",
    root / "robot_calib_result.npz",
    root / "franka_fingertip_table_calibration_simple.json",
    root / "fr3_home_state.npz",
]
for i in range(3):
    required.extend([root / "color" / str(i), root / "depth" / str(i), root / "color" / f"{i}.mp4"])
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing final outputs:\n  " + "\n  ".join(missing))
Path(os.environ["GLOBAL_SUMMARY"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo
echo "================================================================================"
echo "[DONE] Global AnyGrasp-to-lift one-shot complete."
echo "PhysTwin data: $PHYSTWIN_DIR"
echo "Frames:        $FRAME_COUNT"
echo "Videos:        $PHYSTWIN_DIR/color/0.mp4, 1.mp4, 2.mp4"
echo "Camera calib:  $PHYSTWIN_DIR/calibrate.pkl"
echo "Robot calib:   $PHYSTWIN_DIR/robot_calib_result.npz"
echo "Summary:       $GLOBAL_SUMMARY"
echo "================================================================================"
