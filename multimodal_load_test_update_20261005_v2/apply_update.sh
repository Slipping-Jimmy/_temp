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
    if [[ ! -f "$PAYLOAD_DIR/$required" ]]; then
        echo "Update package is incomplete; missing payload/$required" >&2
        exit 2
    fi
done

backup_dir="$TARGET_DIR/_update_backup_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$backup_dir/tests"

files=(
    locustfile.py
    loadtest_lib.py
    run_test.sh
    .env.example
    README.md
    tests/test_loadtest_lib.py
)

for relative in "${files[@]}"; do
    if [[ -f "$TARGET_DIR/$relative" ]]; then
        mkdir -p "$backup_dir/$(dirname "$relative")"
        cp -p "$TARGET_DIR/$relative" "$backup_dir/$relative"
    fi
    mkdir -p "$TARGET_DIR/$(dirname "$relative")"
    cp -p "$PAYLOAD_DIR/$relative" "$TARGET_DIR/$relative"
done
chmod +x "$TARGET_DIR/run_test.sh"

if [[ -f "$TARGET_DIR/.env" ]]; then
    cp -p "$TARGET_DIR/.env" "$backup_dir/.env"

    upsert_env() {
        key="$1"
        value="$2"
        if grep -q "^${key}=" "$TARGET_DIR/.env"; then
            sed -i "s|^${key}=.*|${key}=${value}|" "$TARGET_DIR/.env"
        else
            printf '\n%s=%s\n' "$key" "$value" >> "$TARGET_DIR/.env"
        fi
    }

    upsert_env CONVERSATION_MODE per_user
    upsert_env REQUIRE_DONE_MARKER 1
    if ! grep -q '^FIXED_CONVERSATION_ID=' "$TARGET_DIR/.env"; then
        printf 'FIXED_CONVERSATION_ID=\n' >> "$TARGET_DIR/.env"
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
    "$python_bin" -m py_compile "$TARGET_DIR/locustfile.py" "$TARGET_DIR/loadtest_lib.py"
    (
        cd "$TARGET_DIR"
        PYTHONPATH="$TARGET_DIR" "$python_bin" -m unittest discover -s tests -p 'test_loadtest_lib.py'
    )
else
    echo "Warning: Python not found; skipped validation." >&2
fi

echo
echo "Update installed successfully."
echo "Backup: $backup_dir"
echo "Preserved: .env token, data/, results/, .venv-loadtest/"
echo
echo "Next:"
echo "  cd $TARGET_DIR"
echo "  source .venv-loadtest/bin/activate"
echo "  ./run_test.sh ./.env --smoke"
