#!/bin/bash
set -e

declare -a users=(1 2 3 5 8 10 12 15 20)

IMAGE="locust-ivan"
SCRIPT="locust_test_rag_sse_2.py"

RESULT_DIR="results_new/gemma-3-27b-L40S-rag"
mkdir -p "$RESULT_DIR"

TOKEN_OUTPUT="2gpu_1p_token_times.csv"
SUMMARY_OUTPUT="rag_sse_summary.csv"

GPU_NUM="2gpu"
SPAWN_RATE="1r"

for u in "${users[@]}"; do
  echo "Running locust test with ${u} users..."

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

  mv "$TOKEN_OUTPUT" "${RESULT_DIR}/${GPU_NUM}_${u}p_${SPAWN_RATE}_token_times.csv"
  mv "$SUMMARY_OUTPUT" "${RESULT_DIR}/${GPU_NUM}_${u}p_${SPAWN_RATE}_summary.csv"

  echo "Test completed for ${u} users."
  echo "Token report saved to ${RESULT_DIR}/${GPU_NUM}_${u}p_${SPAWN_RATE}_token_times.csv"
  echo "Summary saved to ${RESULT_DIR}/${GPU_NUM}_${u}p_${SPAWN_RATE}_summary.csv"
done
