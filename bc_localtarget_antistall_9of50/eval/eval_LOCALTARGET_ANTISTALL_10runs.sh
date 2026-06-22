#!/usr/bin/env bash
# Sanity evaluation script: BC LocalTarget Anti-Stall policy, 10 runs
# Policy: bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch
# Task:   Isaac-Stack-5Cube-Franka-IK-Rel-v0
# Date run: 2026-06-19
# Results: 2/10 successes (20%), successful run IDs: 2, 3
#
# Usage (from repo root, inside IsaacLab Python env):
#   bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_10runs.sh
#
# Prerequisites:
#   1. IsaacLab installed and activated (see docs/environment_expected.md)
#   2. franka_5cube_stack extension installed: pip install -e source/franka_5cube_stack
#   3. best.pt present: bc_localtarget_antistall_9of50/policy/best.pt (Git LFS)
#   4. Scripts present: bc_localtarget_antistall_9of50/scripts/play_bc_rnn_target_conditioned.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

POLICY_CHECKPOINT="${REPO_ROOT}/bc_localtarget_antistall_9of50/policy/best.pt"
PLAY_SCRIPT="${REPO_ROOT}/bc_localtarget_antistall_9of50/scripts/play_bc_rnn_target_conditioned.py"
TASK="Isaac-Stack-5Cube-Franka-IK-Rel-v0"
NUM_RUNS=10
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SCRIPT_DIR}/localtarget_antistall_10x_eval_${TIMESTAMP}"

if [ ! -f "${POLICY_CHECKPOINT}" ]; then
    echo "ERROR: Policy checkpoint not found at ${POLICY_CHECKPOINT}"
    echo "       Run: git lfs pull   (to fetch the checkpoint from LFS)"
    exit 1
fi

if [ ! -f "${PLAY_SCRIPT}" ]; then
    echo "ERROR: Play script not found at ${PLAY_SCRIPT}"
    echo "       Copy it from your IsaacLab scripts/custom/ directory."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo " BC LocalTarget Anti-Stall Sanity Eval"
echo " Task:       ${TASK}"
echo " Checkpoint: ${POLICY_CHECKPOINT}"
echo " Runs: ${NUM_RUNS}"
echo " Output: ${OUTPUT_DIR}"
echo "========================================"

for i in $(seq 0 $((NUM_RUNS - 1))); do
    echo "[Run ${i}/${NUM_RUNS}] Starting..."
    python "${PLAY_SCRIPT}" \
        --task "${TASK}" \
        --checkpoint "${POLICY_CHECKPOINT}" \
        --run_id "${i}" \
        --num_envs 1 \
        --headless \
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
