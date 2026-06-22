# test_stacking

Franka 5-cube stacking experiments in IsaacLab — Behavior Cloning and RL policies.

---

## BC LocalTarget Anti-Stall Policy

**Package:** `bc_localtarget_antistall_9of50/`  
**Main result:** **9 / 50 successful stacking episodes (18%)**

### What the policy does

This is a target-conditioned Behavior Cloning policy for Franka 5-cube stacking.

The working recipe uses:
- **Target-conditioned BC** — the policy receives the current target cube index as input
- **LocalTarget representation** — local coordinate frame for the target cube
- **RNN policy checkpoint** — recurrent network for temporal dependencies
- **Anti-stall / guard layer** — detects and recovers from gripper stall conditions
- **Planner-forced discrete gripper** — binary open/close via planner override
- **7D action space:** `dx, dy, dz, rx, ry, rz, gripper`

### Checkpoint

```
bc_localtarget_antistall_9of50/policy/best.pt
```

Source: `IsaacLab/logs/bc_detector_expert/bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch/best.pt`  
Tracked via **Git LFS** (`*.pt`).

### Datasets

```
bc_localtarget_antistall_9of50/datasets/*.hdf5
```

Training demonstration datasets for the 60-demo target-conditioned BC policy.  
Tracked via **Git LFS** (`*.hdf5`).

### Scripts

```
bc_localtarget_antistall_9of50/scripts/
```

Required train / play / eval / source scripts.

### Evaluation Logs

```
bc_localtarget_antistall_9of50/eval/
├── localtarget_antistall_50x_eval_20260619_175528/   ← Main 50-run eval
└── localtarget_antistall_10x_eval_20260619_162950/   ← Sanity 10-run eval
```

### Results

**Main evaluation — 50 runs (2026-06-19 17:55:28):**

| Metric | Value |
|--------|-------|
| Total runs | 50 |
| Successful runs | **9** |
| Success rate | **18%** |
| Successful run IDs | 4, 5, 9, 22, 31, 33, 36, 43, 49 |

**Sanity evaluation — 10 runs (2026-06-19 16:29:50):**

| Metric | Value |
|--------|-------|
| Total runs | 10 |
| Successful runs | **2** |
| Success rate | **20%** |
| Successful run IDs | 2, 3 |

### How to count successes

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=True" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
grep -R "FINAL PHYSICAL STACK SUCCESS=False" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

### List successful runs

```bash
grep -Rl "FINAL PHYSICAL STACK SUCCESS=True" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | sort
```

### Inspect successful logs

```bash
grep -n "FINAL PHYSICAL STACK SUCCESS\|TARGET BC STEP\|guard=\|xy_gap\|z_gap" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_4.log | tail -120
```

> **Note:** Videos are intentionally not committed to this repository.
> Evaluation evidence is preserved as `.log` files only.

See [`bc_localtarget_antistall_9of50/README_BC_9OF50.md`](bc_localtarget_antistall_9of50/README_BC_9OF50.md) for full documentation.

---

