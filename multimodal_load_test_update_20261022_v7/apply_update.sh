#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD_DIR="$SCRIPT_DIR/payload"
TARGET_DIR="${1:-}"

if [[ -z "$TARGET_DIR" ]]; then
    echo "Usage: $0 /path/to/multimodal_load_test" >&2
    exit 2
fi
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

for required in unthrottled_load.py loadtest_lib.py; do
    if [[ ! -f "$TARGET_DIR/$required" ]]; then
        echo "Target does not look like multimodal_load_test; missing: $TARGET_DIR/$required" >&2
        exit 2
    fi
done

files=(
    .env.example
    README.md
    unthrottled_load.py
    inspect_sse.py
    tests/test_unthrottled_load.py
    tests/test_inspect_sse.py
    tests/test_llm_argument.py
)
for relative in "${files[@]}"; do
    if [[ ! -f "$PAYLOAD_DIR/$relative" ]]; then
        echo "Update package is incomplete; missing payload/$relative" >&2
        exit 2
    fi
done

backup_dir="$TARGET_DIR/_update_backup_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$backup_dir/tests"
for relative in "${files[@]}"; do
    if [[ -f "$TARGET_DIR/$relative" ]]; then
        mkdir -p "$backup_dir/$(dirname "$relative")"
        cp -p "$TARGET_DIR/$relative" "$backup_dir/$relative"
    fi
    mkdir -p "$TARGET_DIR/$(dirname "$relative")"
    cp -p "$PAYLOAD_DIR/$relative" "$TARGET_DIR/$relative"
done
chmod +x "$TARGET_DIR/unthrottled_load.py"

python_bin=""
if [[ -x "$TARGET_DIR/.venv-loadtest/bin/python" ]]; then
    python_bin="$TARGET_DIR/.venv-loadtest/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
fi

if [[ -n "$python_bin" ]]; then
    "$python_bin" -m py_compile "$TARGET_DIR/unthrottled_load.py"
    (
        cd "$TARGET_DIR"
        PYTHONPATH="$TARGET_DIR" "$python_bin" -m unittest discover -s tests -p 'test_*.py'
    )
else
    echo "Warning: Python not found; skipped validation." >&2
fi

echo
echo "Update installed successfully."
echo "Backup: $backup_dir"
echo "Preserved: .env, data/, results/, .venv-loadtest/"
echo
echo "REQUIRED before sse-burst: edit $TARGET_DIR/.env and set:"
echo "  FIXED_CONVERSATION_ID="
echo "  UNTHROTTLED_CONVERSATION_STRATEGY=per_request"
echo "  SSE_MAX_TOTAL_SECONDS=180"
echo "  CAPTURE_RESPONSE=1"
echo "LLM_ARGUMENT_JSON defaults to Gemma 4 sampling + max_tokens=1024 + repetition_penalty=1.05."
echo "Confirm the backend accepts and forwards top_k/repetition_penalty and applies enable_thinking."
echo
echo "Then run:"
echo "  cd $TARGET_DIR"
echo "  source .venv-loadtest/bin/activate"
echo "  python3 unthrottled_load.py --config ./.env --output-root ./results sse-burst --concurrency '5 10 20' --rounds 3 --between-rounds 20s --prepare-workers 8"

