#!/bin/bash
set -e

declare -a users=(1 2 3 5 8 10 12 15 20)

IMAGE="locust-ivan"
SCRIPT="locust_test_vllm.py"

TOKEN_OUTPUT="2gpu_1p_token_times.csv"
SUMMARY_OUTPUT="vllm_summary.csv"

RESULT_DIR="./results_new/gemma-3-27b-L40S-vllm"

mkdir -p "$RESULT_DIR"

for u in "${users[@]}"; do
    echo "======================================="
    echo "Running vLLM locust test with ${u} users"
    echo "======================================="

    docker run --rm \
        -v "$(pwd):/workspace" \
        -w /workspace \
        "$IMAGE" \
        locust \
        -f "$SCRIPT" \
        --headless \
        -u "$u" \
        -r 1 \
        --run-time 5m

    mv "$TOKEN_OUTPUT" \
       "${RESULT_DIR}/2gpu_${u}p_token_times.csv"

    mv "$SUMMARY_OUTPUT" \
       "${RESULT_DIR}/2gpu_${u}p_summary.csv"

    echo "Completed ${u} users"
    echo
done

echo "======================================="
echo "All tests completed"
echo "Results saved to:"
echo "$RESULT_DIR"
echo "======================================="
