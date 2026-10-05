#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-$SCRIPT_DIR/.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Configuration not found: $CONFIG_FILE" >&2
    echo "Copy .env.example to .env and edit it first." >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_FILE"
set +a

if [[ "${2:-}" == "--smoke" ]]; then
    USERS_STEPS="1"
    RUN_TIME="1m"
fi

cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

: "${PLAYGROUND_BASE_URL:?PLAYGROUND_BASE_URL is required}"
: "${PLAYGROUND_TOKEN:?PLAYGROUND_TOKEN is required}"
: "${DATASET_DIR:=./data/tono}"
: "${IMAGE_MANIFEST:=./images_manifest.csv}"
: "${QUESTION_FILE:=./questions.json}"
: "${PROMPT_PROFILE:=short}"
: "${CONVERSATION_MODE:=per_user}"
: "${FIXED_CONVERSATION_ID:=}"
: "${USERS_STEPS:=1 2 4 8 16 32}"
: "${SPAWN_RATE:=2}"
: "${RUN_TIME:=5m}"
: "${STOP_TIMEOUT:=30}"
: "${METRICS_INTERVAL_SECONDS:=1}"
: "${COLLECT_LOCAL_GPU:=0}"

if [[ "$PLAYGROUND_TOKEN" == replace-* ]]; then
    echo "Replace PLAYGROUND_TOKEN with a new test token." >&2
    exit 2
fi
if ! command -v locust >/dev/null 2>&1; then
    echo "locust is not installed. Run: python3 -m pip install -r requirements.txt" >&2
    exit 2
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/results/$RUN_ID}"
STAGE_FILE="$RESULT_ROOT/current_stage.txt"
mkdir -p "$RESULT_ROOT"
export RUN_ID RESULT_ROOT

if [[ ! -f "$IMAGE_MANIFEST" ]]; then
    echo "Building image manifest..."
    set +e
    python3 "$SCRIPT_DIR/build_manifest.py" "$DATASET_DIR" --output "$IMAGE_MANIFEST"
    manifest_exit=$?
    set -e
    if [[ "$manifest_exit" -ne 0 ]]; then
        echo "Manifest builder reported invalid images; valid entries will still be used." >&2
    fi
fi
if [[ ! -s "$IMAGE_MANIFEST" ]]; then
    echo "Image manifest is missing or empty: $IMAGE_MANIFEST" >&2
    exit 2
fi

cp "$IMAGE_MANIFEST" "$RESULT_ROOT/images_manifest.csv"
cp "$QUESTION_FILE" "$RESULT_ROOT/questions.json"

{
    echo "RUN_ID=$RUN_ID"
    echo "PLAYGROUND_BASE_URL=$PLAYGROUND_BASE_URL"
    echo "DATASET_DIR=$DATASET_DIR"
    echo "IMAGE_MANIFEST=$IMAGE_MANIFEST"
    echo "QUESTION_FILE=$QUESTION_FILE"
    echo "PROMPT_PROFILE=$PROMPT_PROFILE"
    echo "CONVERSATION_MODE=$CONVERSATION_MODE"
    echo "FIXED_CONVERSATION_ID_SET=$([[ -n "$FIXED_CONVERSATION_ID" ]] && echo 1 || echo 0)"
    echo "USERS_STEPS=$USERS_STEPS"
    echo "SPAWN_RATE=$SPAWN_RATE"
    echo "RUN_TIME=$RUN_TIME"
    echo "STOP_TIMEOUT=$STOP_TIMEOUT"
    echo "REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-120}"
    echo "BETWEEN_REQUESTS_SECONDS=${BETWEEN_REQUESTS_SECONDS:-0}"
    echo "REQUIRE_DONE_MARKER=${REQUIRE_DONE_MARKER:-1}"
    echo "VLLM_METRICS_URL=${VLLM_METRICS_URL:-}"
    echo "COLLECT_LOCAL_GPU=$COLLECT_LOCAL_GPU"
    echo "MAX_ERROR_RATE_PERCENT=${MAX_ERROR_RATE_PERCENT:-1}"
    echo "MAX_P95_TTFT_MS=${MAX_P95_TTFT_MS:-3000}"
    echo "MAX_P95_TOTAL_MS=${MAX_P95_TOTAL_MS:-10000}"
    echo "hostname=$(hostname)"
    echo "python=$(python3 --version 2>&1)"
    echo "locust=$(locust --version 2>&1)"
} > "$RESULT_ROOT/run_config.txt"

MONITOR_PID=""
cleanup() {
    printf '%s\n' "idle" > "$STAGE_FILE"
    if [[ -n "$MONITOR_PID" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

MONITOR_ARGS=(
    --output-dir "$RESULT_ROOT/monitor"
    --interval "$METRICS_INTERVAL_SECONDS"
    --stage-file "$STAGE_FILE"
)
if [[ -n "${VLLM_METRICS_URL:-}" ]]; then
    MONITOR_ARGS+=(--vllm-url "$VLLM_METRICS_URL")
fi
if [[ "$COLLECT_LOCAL_GPU" == "1" ]]; then
    MONITOR_ARGS+=(--gpu)
fi
if [[ -n "${VLLM_METRICS_URL:-}" || "$COLLECT_LOCAL_GPU" == "1" ]]; then
    python3 "$SCRIPT_DIR/collect_metrics.py" "${MONITOR_ARGS[@]}" &
    MONITOR_PID=$!
fi

stage_number=0
for users in $USERS_STEPS; do
    if ! [[ "$users" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid user count in USERS_STEPS: $users" >&2
        exit 2
    fi
    stage_number=$((stage_number + 1))
    stage_name="$(printf 'stage_%03d_%su' "$stage_number" "$users")"
    stage_dir="$RESULT_ROOT/$stage_name"
    mkdir -p "$stage_dir"
    printf '%s\n' "$stage_name" > "$STAGE_FILE"
    export RESULT_DIR="$stage_dir"
    export TARGET_USERS="$users"

    echo "[$(date -u +%FT%TZ)] Starting $stage_name for $RUN_TIME"
    date -u +%s.%N > "$stage_dir/stage_start_epoch.txt"
    set +e
    locust \
        -f "$SCRIPT_DIR/locustfile.py" \
        --headless \
        --host "$PLAYGROUND_BASE_URL" \
        -u "$users" \
        -r "$SPAWN_RATE" \
        --run-time "$RUN_TIME" \
        --stop-timeout "$STOP_TIMEOUT" \
        --csv "$stage_dir/locust" \
        --csv-full-history \
        --html "$stage_dir/locust_report.html" \
        --logfile "$stage_dir/locust.log"
    exit_code=$?
    set -e
    date -u +%s.%N > "$stage_dir/stage_end_epoch.txt"
    printf '%s\n' "$exit_code" > "$stage_dir/locust_exit_code.txt"
    echo "[$(date -u +%FT%TZ)] Finished $stage_name (Locust exit $exit_code)"
done

printf '%s\n' "analysis" > "$STAGE_FILE"
python3 "$SCRIPT_DIR/analyze_results.py" "$RESULT_ROOT"
cleanup
trap - EXIT INT TERM

echo "Results are ready: $RESULT_ROOT"
