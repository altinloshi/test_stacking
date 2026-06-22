# BC LocalTarget Anti-Stall Policy — 9/50 Evaluation

**Task:** Franka 5-cube stacking (simulation)  
**Policy type:** Target-conditioned Behavior Cloning (BC) with RNN  
**Checkpoint:** `bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch`  
**Main result:** **9 / 50 successful stacking episodes (18%)**

---

## Policy Description

This is a target-conditioned Behavior Cloning policy for Franka 5-cube stacking.

The working recipe uses:
- **Target-conditioned BC** — the policy receives the current target cube index as input
- **LocalTarget representation** — local coordinate frame for the target cube
- **RNN policy checkpoint** — recurrent network for temporal dependencies
- **Anti-stall / guard layer** — detects and recovers from gripper stall conditions
- **Planner-forced discrete gripper** — binary open/close gripper via planner override
- **7D action space:** `dx, dy, dz, rx, ry, rz, gripper`

---

## Package Structure

```
bc_localtarget_antistall_9of50/
├── policy/
│   └── best.pt                  ← Main RNN BC policy checkpoint (Git LFS)
├── datasets/
│   └── *.hdf5                   ← Training demonstration datasets (Git LFS)
├── scripts/
│   └── *.py                     ← Train / play / eval / source scripts
├── eval/
│   ├── localtarget_antistall_50x_eval_20260619_175528/   ← Main 50-run eval logs
│   ├── localtarget_antistall_10x_eval_20260619_162950/   ← Sanity 10-run eval logs
│   ├── eval_LOCALTARGET_ANTISTALL_50runs.sh
│   └── eval_LOCALTARGET_ANTISTALL_10runs.sh
├── logs/
│   └── *.log                    ← Training / collection logs
└── docs/
    ├── package_manifest.txt
    ├── file_sources.txt
    └── git_status_snapshot.txt
```

---

## Checkpoint

| Field | Value |
|-------|-------|
| File | `bc_localtarget_antistall_9of50/policy/best.pt` |
| Source | `IsaacLab/logs/bc_detector_expert/bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch/best.pt` |
| Format | PyTorch checkpoint (tracked via Git LFS) |

---

## Datasets

Located in `bc_localtarget_antistall_9of50/datasets/`.

All `.hdf5` files are tracked via Git LFS. These are the demonstration datasets used to train the 60-demo target-conditioned BC policy.

---

## Scripts

Located in `bc_localtarget_antistall_9of50/scripts/`.

Includes the required train / play / eval / source scripts for this policy.

---

## Evaluation Results

### Main Evaluation — 50 runs (2026-06-19 17:55:28)

| Metric | Value |
|--------|-------|
| Total runs | 50 |
| **Successful runs** | **9** |
| **Success rate** | **18%** |
| Successful run IDs | 4, 5, 9, 22, 31, 33, 36, 43, 49 |

Logs: `bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/`

### Sanity Evaluation — 10 runs (2026-06-19 16:29:50)

| Metric | Value |
|--------|-------|
| Total runs | 10 |
| **Successful runs** | **2** |
| **Success rate** | **20%** |
| Successful run IDs | 2, 3 |

Logs: `bc_localtarget_antistall_9of50/eval/localtarget_antistall_10x_eval_20260619_162950/`

> **Note:** Videos are intentionally not committed to this repository.
> The evaluation evidence is preserved as `.log` files only.

---

## How to Count Successes

Count successful runs in the main 50-run evaluation:

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=True" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

Count failed runs:

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=False" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

---

## List Successful Runs

```bash
grep -Rl "FINAL PHYSICAL STACK SUCCESS=True" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | sort
```

---

## Inspect Successful Logs

Inspect key events in a successful run (e.g., run 4):

```bash
grep -n "FINAL PHYSICAL STACK SUCCESS\|TARGET BC STEP\|guard=\|xy_gap\|z_gap" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_4.log | tail -120
```

---

## Reproduce Evaluation

Run 50-episode evaluation:

```bash
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_50runs.sh
```

Run 10-episode sanity evaluation:

```bash
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_10runs.sh
```

---

## Git LFS

Large binary files are tracked via Git LFS:

| Pattern | Type |
|---------|------|
| `*.pt` | PyTorch model checkpoints |
| `*.pth` | PyTorch state dicts |
| `*.hdf5` | HDF5 demonstration datasets |

To verify LFS-tracked files:

```bash
git lfs ls-files
```

---

## Notes

- Videos are **intentionally not committed**. Evaluation evidence is log-based only.
- The anti-stall guard is a key component that distinguishes this working recipe from earlier failed attempts.
- The planner-forced discrete gripper (binary open/close) was critical for reliable stacking.
- Successful runs: 4, 5, 9, 22, 31, 33, 36, 43, 49 (main eval); 2, 3 (sanity eval).
