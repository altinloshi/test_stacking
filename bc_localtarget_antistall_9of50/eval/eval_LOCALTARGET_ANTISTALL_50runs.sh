#!/usr/bin/env bash
# Evaluation script: BC LocalTarget Anti-Stall policy, 50 runs
# Policy: bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch
# Date run: 2026-06-19
# Results: 9/50 successes (18%), successful run IDs: 4, 5, 9, 22, 31, 33, 36, 43, 49

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

POLICY_CHECKPOINT="${REPO_ROOT}/bc_localtarget_antistall_9of50/policy/best.pt"
NUM_RUNS=50
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SCRIPT_DIR}/localtarget_antistall_50x_eval_${TIMESTAMP}"

if [ ! -f "${POLICY_CHECKPOINT}" ]; then
    echo "ERROR: Policy checkpoint not found at ${POLICY_CHECKPOINT}"
    echo "       Please ensure best.pt is present (tracked via Git LFS)."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo " BC LocalTarget Anti-Stall Evaluation"
echo " Checkpoint: ${POLICY_CHECKPOINT}"
echo " Runs: ${NUM_RUNS}"
echo " Output: ${OUTPUT_DIR}"
echo "========================================"

for i in $(seq 0 $((NUM_RUNS - 1))); do
    echo "[Run ${i}/${NUM_RUNS}] Starting..."
    python "${REPO_ROOT}/bc_localtarget_antistall_9of50/scripts/play_bc_localtarget.py" \
        --checkpoint "${POLICY_CHECKPOINT}" \
        --run_id "${i}" \
        --output_dir "${OUTPUT_DIR}" \
        > "${OUTPUT_DIR}/run_${i}.log" 2>&1
    
    if grep -q "FINAL PHYSICAL STACK SUCCESS=True" "${OUTPUT_DIR}/run_${i}.log"; then
        echo "[Run ${i}] SUCCESS"
    else
        echo "[Run ${i}] FAILED"
    fi
done

echo ""
echo "======== EVALUATION COMPLETE ========"
SUCCESSES=$(grep -R "FINAL PHYSICAL STACK SUCCESS=True" "${OUTPUT_DIR}"/run_*.log | wc -l)
FAILURES=$(grep -R "FINAL PHYSICAL STACK SUCCESS=False" "${OUTPUT_DIR}"/run_*.log | wc -l)
echo "Successes: ${SUCCESSES}/${NUM_RUNS}"
echo "Failures:  ${FAILURES}/${NUM_RUNS}"
echo "Output:    ${OUTPUT_DIR}"
echo ""
echo "Successful run IDs:"
grep -Rl "FINAL PHYSICAL STACK SUCCESS=True" "${OUTPUT_DIR}"/run_*.log | sort
