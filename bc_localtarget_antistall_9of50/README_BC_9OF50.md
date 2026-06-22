# BC LocalTarget Anti-Stall Policy — Franka 5-Cube Stacking

This folder contains the reproducibility package for the Franka 5-cube stacking Behavior Cloning policy.

The goal of this package is to let another engineer clone the repository, pull the trained policy and dataset files, inspect the evaluation evidence, and rerun the policy inside a matching IsaacLab environment.

---

## Result Summary

Policy type:

- Target-conditioned Behavior Cloning
- LocalTarget representation
- RNN policy checkpoint
- Anti-stall / guard layer
- Planner-forced discrete gripper
- 7D action space: `dx, dy, dz, rx, ry, rz, gripper`

Main evaluation:

- 50 single-env rollouts
- 9 successful runs
- 41 failed runs
- Success rate: 18%

Successful 50-run IDs:

```text
4, 5, 9, 22, 31, 33, 36, 43, 49
```

Sanity evaluation (earlier 10-run check):

- 10 single-env rollouts
- 2 successful runs
- Success rate: 20%

Successful 10-run IDs:

```text
2, 3
```

> Videos are intentionally not committed. Evaluation evidence is preserved as `.log` files only.

---

## Package Structure

```
bc_localtarget_antistall_9of50/
├── policy/
│   └── best.pt                    ← Trained RNN BC checkpoint (Git LFS)
├── datasets/
│   └── *.hdf5                     ← Demonstration datasets used for training (Git LFS)
├── scripts/
│   └── *.py                       ← Train / play / eval / source scripts
├── eval/
│   ├── localtarget_antistall_50x_eval_20260619_175528/
│   │   └── run_0.log … run_49.log ← Main 50-run evaluation logs
│   ├── localtarget_antistall_10x_eval_20260619_162950/
│   │   └── run_0.log … run_9.log  ← Sanity 10-run evaluation logs
│   ├── eval_LOCALTARGET_ANTISTALL_50runs.sh
│   └── eval_LOCALTARGET_ANTISTALL_10runs.sh
├── logs/
│   └── *.log                      ← Training / collection logs
├── docs/
│   ├── package_manifest.txt
│   ├── file_sources.txt
│   └── git_status_snapshot.txt
└── README_BC_9OF50.md             ← This file
```

---

## Checkpoint

| Field | Value |
|-------|-------|
| File | `bc_localtarget_antistall_9of50/policy/best.pt` |
| Original path | `IsaacLab/logs/bc_detector_expert/bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch/best.pt` |
| Format | PyTorch checkpoint |
| Git tracking | LFS (`*.pt`) |

---

## Datasets

Located in `bc_localtarget_antistall_9of50/datasets/`.

These are the HDF5 demonstration files used to train the 60-demo target-conditioned BC policy. All `.hdf5` files are tracked via Git LFS.

---

## Scripts

Located in `bc_localtarget_antistall_9of50/scripts/`.

Contains the required train, play, and evaluation scripts for this policy. Do not add unrelated RL or PPO scripts to this folder.

---

## Evaluation Logs

### Main evaluation — 50 runs

Directory: `bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/`

| Metric | Value |
|--------|-------|
| Total runs | 50 |
| Successful runs | **9** |
| Failed runs | 41 |
| Success rate | **18%** |
| Successful run IDs | 4, 5, 9, 22, 31, 33, 36, 43, 49 |

### Sanity evaluation — 10 runs

Directory: `bc_localtarget_antistall_9of50/eval/localtarget_antistall_10x_eval_20260619_162950/`

| Metric | Value |
|--------|-------|
| Total runs | 10 |
| Successful runs | **2** |
| Failed runs | 8 |
| Success rate | **20%** |
| Successful run IDs | 2, 3 |

---

## How to Count Successes

Count successes in the main 50-run evaluation:

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=True" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

Count failures:

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=False" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

---

## List Successful Runs

```bash
grep -Rl "FINAL PHYSICAL STACK SUCCESS=True" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | sort
```

---

## Inspect a Successful Log

Inspect key events in run 4 (successful):

```bash
grep -n "FINAL PHYSICAL STACK SUCCESS\|TARGET BC STEP\|guard=\|xy_gap\|z_gap" bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_4.log | tail -120
```

---

## Reproduce the Evaluation

Run the 50-episode evaluation:

```bash
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_50runs.sh
```

Run the 10-episode sanity evaluation:

```bash
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_10runs.sh
```

---

## Git LFS

Large binary files are tracked via Git LFS. To pull them after cloning:

```bash
git lfs pull
```

Tracked extensions:

| Pattern | Type |
|---------|------|
| `*.pt` | PyTorch model checkpoints |
| `*.pth` | PyTorch state dicts |
| `*.hdf5` | HDF5 demonstration datasets |

Verify LFS-tracked files:

```bash
git lfs ls-files
```

---

## Notes

- Videos are **intentionally not committed**. Evaluation evidence is log-based only.
- The anti-stall guard is the key component that distinguishes this working recipe from earlier failed attempts.
- The planner-forced discrete gripper (binary open/close) was critical for reliable stacking behaviour.
- Earlier attempts without LocalTarget or anti-stall failed to achieve consistent success.
