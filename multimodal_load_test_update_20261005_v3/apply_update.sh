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

for required in locustfile.py loadtest_lib.py run_test.sh; do
    if [[ ! -f "$TARGET_DIR/$required" ]]; then
        echo "Target does not look like multimodal_load_test; missing: $TARGET_DIR/$required" >&2
        exit 2
    fi
done

files=(
    .env.example
    README.md
    loadtest_lib.py
    locustfile.py
    run_test.sh
    unthrottled_load.py
    tests/test_loadtest_lib.py
    tests/test_unthrottled_load.py
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
chmod +x "$TARGET_DIR/run_test.sh" "$TARGET_DIR/unthrottled_load.py"

if [[ -f "$TARGET_DIR/.env" ]]; then
    cp -p "$TARGET_DIR/.env" "$backup_dir/.env"
    if ! grep -q '^UNTHROTTLED_CONVERSATION_STRATEGY=' "$TARGET_DIR/.env"; then
        printf '\n# Used by unthrottled_load.py; keeps one reusable conversation by default.\n' >> "$TARGET_DIR/.env"
        printf 'UNTHROTTLED_CONVERSATION_STRATEGY=shared\n' >> "$TARGET_DIR/.env"
    fi
else
    echo "Warning: $TARGET_DIR/.env does not exist; copy .env.example and add a fresh token." >&2
fi

python_bin=""
if [[ -x "$TARGET_DIR/.venv-loadtest/bin/python" ]]; then
    python_bin="$TARGET_DIR/.venv-loadtest/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
fi

if [[ -n "$python_bin" ]]; then
    "$python_bin" -m py_compile \
        "$TARGET_DIR/locustfile.py" \
        "$TARGET_DIR/loadtest_lib.py" \
        "$TARGET_DIR/unthrottled_load.py"
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
echo "Preserved: .env token, data/, results/, .venv-loadtest/"
echo
echo "Next (same-token unthrottled burst):"
echo "  cd $TARGET_DIR"
echo "  source .venv-loadtest/bin/activate"
echo "  python3 unthrottled_load.py --config ./.env --output-root ./results burst --concurrency '8 16 24' --rounds 3 --between-rounds 20s"
