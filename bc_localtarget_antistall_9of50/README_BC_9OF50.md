# BC LocalTarget Anti-Stall Policy — 9/50 Evaluation

**Task:** Franka 5-cube stacking (simulation, IsaacLab)
**Policy type:** Target-conditioned Behavior Cloning (BC) with RNN
**Checkpoint:** `bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch`
**Main result:** **9 / 50 successful stacking episodes (18%)**

---

## Table of Contents

1. [Policy Description](#policy-description)
2. [Package Structure](#package-structure)
3. [Quick Start — Verify Saved Results](#quick-start--verify-saved-results)
4. [Setup Environment](#setup-environment)
5. [Fetch Large Files (Git LFS)](#fetch-large-files-git-lfs)
6. [Add Missing Files from Local Machine](#add-missing-files-from-local-machine)
7. [Run Fresh Evaluation](#run-fresh-evaluation)
8. [Inspect Successful Logs](#inspect-successful-logs)
9. [Train from Scratch](#train-from-scratch)
10. [Collect Demonstration Data](#collect-demonstration-data)
11. [Evaluation Results](#evaluation-results)
12. [Git LFS Reference](#git-lfs-reference)
13. [Notes](#notes)

---

## Policy Description

This is a **Stage A/B prototype** for Franka 5-cube stacking:

| Stage | Description |
|-------|-------------|
| **A** | HDF5 demonstration data from scripted StackPlanner / expert controller |
| **B** | Target-conditioned BC RNN policy with LocalTarget representation, anti-stall guard, and planner-forced gripper |
| **C** | RL fine-tuning with domain randomization — **not completed** |

The working recipe uses:
- **Target-conditioned BC** — the policy receives the current target cube index as input
- **LocalTarget representation** — local coordinate frame relative to the target cube
- **RNN policy checkpoint** — recurrent network for temporal dependencies
- **Anti-stall guard layer** — detects and recovers from gripper stall conditions
- **Planner-forced discrete gripper** — binary open/close via planner override
- **7D action space:** `dx, dy, dz, rx, ry, rz, gripper`
- **Task:** `Isaac-Stack-5Cube-Franka-IK-Rel-v0`
- **Demos:** 60 demonstrations, 20 training epochs

---

## Package Structure

```
bc_localtarget_antistall_9of50/
├── policy/
│   └── best.pt                  ← Main RNN BC policy checkpoint (Git LFS)
├── datasets/
│   └── *.hdf5                   ← Training demonstration datasets (Git LFS)
├── scripts/
│   ├── play_bc_rnn_target_conditioned.py
│   ├── train_bc_rnn_target_conditioned.py
│   ├── run_5cube_camera_hierarchical_success_fixed.py
│   └── (localtarget, antistall, hdf5 utility scripts)
├── eval/
│   ├── localtarget_antistall_50x_eval_20260619_175528/   ← Main 50-run eval logs
│   │   └── run_0.log ... run_49.log
│   ├── localtarget_antistall_10x_eval_20260619_162950/   ← Sanity 10-run eval logs
│   │   └── run_0.log ... run_9.log
│   ├── eval_LOCALTARGET_ANTISTALL_50runs.sh
│   └── eval_LOCALTARGET_ANTISTALL_10runs.sh
├── logs/
│   └── *.log                    ← Training / collection logs
└── docs/
    ├── package_manifest.txt
    ├── file_sources.txt
    ├── git_status_snapshot.txt
    └── environment_expected.md
```

---

## Quick Start — Verify Saved Results

After cloning the repo and pulling LFS files, verify the documented results directly
from the saved eval logs (no IsaacLab required):

```bash
# Clone and fetch LFS files
git clone https://github.com/altinloshi/test_stacking.git
cd test_stacking
git lfs install
git lfs pull
```

Once the eval logs are present (added from local machine, see below), run:

```bash
# Count successful runs in main 50-run eval (expected: 9)
grep -R "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log \
  | wc -l

# Count failed runs (expected: 41)
grep -R "FINAL PHYSICAL STACK SUCCESS=False" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log \
  | wc -l

# List successful run IDs (expected: run_4, run_5, run_9, run_22, run_31, run_33, run_36, run_43, run_49)
grep -Rl "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log \
  | sort

# Count successful runs in sanity 10-run eval (expected: 2)
grep -R "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_10x_eval_20260619_162950/run_*.log \
  | wc -l
```

---

## Setup Environment

IsaacLab must be installed separately. This repo provides the `franka_5cube_stack`
extension and the policy scripts.

### 1. Install IsaacLab

Follow the [IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/).
Activate the IsaacLab Python environment before running any scripts below.

### 2. Clone this repo

```bash
git clone https://github.com/altinloshi/test_stacking.git
cd test_stacking
```

### 3. Register the franka_5cube_stack extension

```bash
pip install -e source/franka_5cube_stack
```

This registers the `Isaac-Stack-5Cube-Franka-IK-Rel-v0` task with IsaacLab.
Run this once in your IsaacLab Python environment.

### 4. Install robomimic (required for BC training)

```bash
pip install robomimic
```

### 5. Verify task registration

```bash
python -c "import gymnasium as gym; import franka_5cube_stack; print(gym.spec('Isaac-Stack-5Cube-Franka-IK-Rel-v0'))"
```

Expected output: `EnvSpec(id='Isaac-Stack-5Cube-Franka-IK-Rel-v0', ...)`

---

## Fetch Large Files (Git LFS)

The policy checkpoint and datasets are stored in Git LFS.

```bash
# Install Git LFS (once per machine)
git lfs install

# Pull all LFS-tracked files
git lfs pull

# Verify LFS files are present
git lfs ls-files

# Check the checkpoint is a valid PyTorch file (not an LFS pointer)
python -c "import torch; ck = torch.load('bc_localtarget_antistall_9of50/policy/best.pt', map_location='cpu'); print('Checkpoint loaded OK, keys:', list(ck.keys())[:5])"
```

LFS-tracked file patterns (defined in `.gitattributes`):

| Pattern | Type |
|---------|------|
| `*.pt`  | PyTorch model checkpoints |
| `*.pth` | PyTorch state dicts |
| `*.hdf5`| HDF5 demonstration datasets |

---

## Add Missing Files from Local Machine

Binary files and eval logs must be added from the local machine.
Follow these steps from the `test_stacking/` repo root on your local machine.

### Add policy checkpoint

```bash
cp /home/a0loshi1/robotics/IsaacLab/logs/bc_detector_expert/bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch/best.pt \
   bc_localtarget_antistall_9of50/policy/best.pt

git add bc_localtarget_antistall_9of50/policy/best.pt
git commit -m "Add BC policy checkpoint best.pt via LFS"
git push
```

### Add evaluation logs (main 50-run eval)

```bash
cp -r /tmp/localtarget_antistall_50x_eval_20260619_175528/ \
   bc_localtarget_antistall_9of50/eval/

git add bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/
git commit -m "Add main 50-run evaluation logs (9/50 successes)"
git push
```

### Add evaluation logs (sanity 10-run eval)

```bash
cp -r /tmp/localtarget_antistall_10x_eval_20260619_162950/ \
   bc_localtarget_antistall_9of50/eval/

git add bc_localtarget_antistall_9of50/eval/localtarget_antistall_10x_eval_20260619_162950/
git commit -m "Add sanity 10-run evaluation logs (2/10 successes)"
git push
```

### Add training datasets (HDF5)

```bash
# Copy relevant HDF5 files from IsaacLab datasets directory
# Only copy files related to: TARGET_CONDITIONED, LOCALTARGET, policyobs, rgbd, stack_demos, robomimic
cp /home/a0loshi1/robotics/IsaacLab/datasets/*TARGET_CONDITIONED*.hdf5 \
   bc_localtarget_antistall_9of50/datasets/ 2>/dev/null || true
cp /home/a0loshi1/robotics/IsaacLab/datasets/*LOCALTARGET*.hdf5 \
   bc_localtarget_antistall_9of50/datasets/ 2>/dev/null || true
cp /home/a0loshi1/robotics/task-franka-cubestack-base/datasets/*policyobs*.hdf5 \
   bc_localtarget_antistall_9of50/datasets/ 2>/dev/null || true

git add bc_localtarget_antistall_9of50/datasets/*.hdf5
git commit -m "Add BC training datasets via LFS"
git push
```

### Add scripts

```bash
# Copy required scripts from IsaacLab custom scripts directory
for f in \
    play_bc_rnn_target_conditioned.py \
    train_bc_rnn_target_conditioned.py \
    run_5cube_camera_hierarchical_success_fixed.py; do
    cp /home/a0loshi1/robotics/IsaacLab/scripts/custom/$f \
       bc_localtarget_antistall_9of50/scripts/ 2>/dev/null || \
    cp /home/a0loshi1/robotics/task-franka-cubestack-base/scripts/$f \
       bc_localtarget_antistall_9of50/scripts/ 2>/dev/null || \
    echo "WARNING: $f not found"
done

# Also copy localtarget, antistall, hdf5 utility scripts
cp /home/a0loshi1/robotics/IsaacLab/scripts/custom/localtarget_*.py \
   bc_localtarget_antistall_9of50/scripts/ 2>/dev/null || true
cp /home/a0loshi1/robotics/IsaacLab/scripts/custom/antistall_*.py \
   bc_localtarget_antistall_9of50/scripts/ 2>/dev/null || true
cp /home/a0loshi1/robotics/IsaacLab/scripts/custom/hdf5_*.py \
   bc_localtarget_antistall_9of50/scripts/ 2>/dev/null || true

git add bc_localtarget_antistall_9of50/scripts/
git commit -m "Add BC policy scripts (play, train, collect, localtarget, antistall)"
git push
```

### Add training logs (optional)

```bash
for f in \
    /tmp/train_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch.log \
    /tmp/play_TARGET_CONDITIONED_LOCALTARGET_700_antistall.log \
    /tmp/play_TARGET_CONDITIONED_LOCALTARGET_700.log \
    /tmp/collect_TARGET_CONDITIONED_BC_50env_700steps.log \
    /tmp/collect_TARGET_BC_batch02.log \
    /tmp/collect_TARGET_BC_batch03.log; do
    [ -f "$f" ] && cp "$f" bc_localtarget_antistall_9of50/logs/
done

git add bc_localtarget_antistall_9of50/logs/*.log 2>/dev/null || true
git commit -m "Add training and collection logs" 2>/dev/null || true
git push
```

---

## Run Fresh Evaluation

Requires IsaacLab environment + franka_5cube_stack + best.pt + play script.

### Run 50-episode evaluation (reproduces the saved result)

```bash
# From repo root, inside IsaacLab Python env:
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_50runs.sh
```

### Run 10-episode sanity evaluation

```bash
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_10runs.sh
```

### Run a single episode directly

```bash
python bc_localtarget_antistall_9of50/scripts/play_bc_rnn_target_conditioned.py \
    --task Isaac-Stack-5Cube-Franka-IK-Rel-v0 \
    --checkpoint bc_localtarget_antistall_9of50/policy/best.pt \
    --run_id 0 \
    --num_envs 1 \
    --headless \
    --output_dir /tmp/bc_test_run/
```

### Run with display (Isaac Sim viewport)

```bash
python bc_localtarget_antistall_9of50/scripts/play_bc_rnn_target_conditioned.py \
    --task Isaac-Stack-5Cube-Franka-IK-Rel-v0 \
    --checkpoint bc_localtarget_antistall_9of50/policy/best.pt \
    --run_id 0 \
    --num_envs 1
```

---

## Inspect Successful Logs

Inspect key events in a successful run (e.g., run 4):

```bash
grep -n "FINAL PHYSICAL STACK SUCCESS\|TARGET BC STEP\|guard=\|xy_gap\|z_gap" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_4.log \
  | tail -120
```

Inspect a failed run for comparison (e.g., run 0):

```bash
grep -n "FINAL PHYSICAL STACK SUCCESS\|TARGET BC STEP\|guard=\|stall\|xy_gap\|z_gap" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_0.log \
  | tail -120
```

Show all successful run IDs from saved logs:

```bash
grep -Rl "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log \
  | sort
```

---

## Train from Scratch

Requires: IsaacLab env, franka_5cube_stack, robomimic, HDF5 demo datasets.

**Note:** The original training used 60 demos and 20 epochs, resulting in `best.pt`.
Do not retrain unless you want to reproduce the full pipeline from scratch.

### 1. Verify datasets are present

```bash
ls bc_localtarget_antistall_9of50/datasets/*.hdf5
```

### 2. Run BC RNN training

```bash
python bc_localtarget_antistall_9of50/scripts/train_bc_rnn_target_conditioned.py \
    --dataset bc_localtarget_antistall_9of50/datasets/ \
    --num_demos 60 \
    --num_epochs 20 \
    --output_dir logs/bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch/ \
    2>&1 | tee /tmp/train_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch.log
```

### 3. Copy the best checkpoint

```bash
cp logs/bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch/best.pt \
   bc_localtarget_antistall_9of50/policy/best.pt
```

---

## Collect Demonstration Data

Requires: IsaacLab env + franka_5cube_stack registered.

**Note:** The original dataset used 60 demonstrations collected by the scripted expert.
Do not recollect unless reproducing the full pipeline.

### Collect via scripted expert (StackPlanner)

```bash
python bc_localtarget_antistall_9of50/scripts/run_5cube_camera_hierarchical_success_fixed.py \
    --task Isaac-Stack-5Cube-Franka-IK-Rel-v0 \
    --num_envs 50 \
    --num_steps 700 \
    --output_dir bc_localtarget_antistall_9of50/datasets/ \
    --headless \
    2>&1 | tee /tmp/collect_TARGET_CONDITIONED_BC_50env_700steps.log
```

### Verify collected HDF5 files

```bash
python -c "
import h5py, sys
for f in sys.argv[1:]:
    with h5py.File(f, 'r') as h:
        demos = list(h.get('data', h).keys())
        print(f'{f}: {len(demos)} demos')
" bc_localtarget_antistall_9of50/datasets/*.hdf5
```

---

## Evaluation Results

### Main Evaluation — 50 runs (2026-06-19 17:55:28)

| Metric | Value |
|--------|-------|
| Total runs | 50 |
| **Successful runs** | **9** |
| **Success rate** | **18%** |
| Failed runs | 41 |
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
> Evaluation evidence is preserved as `.log` files only.

---

## Git LFS Reference

All large binary files are tracked via Git LFS:

| Pattern | Type |
|---------|------|
| `*.pt`  | PyTorch model checkpoints |
| `*.pth` | PyTorch state dicts |
| `*.hdf5`| HDF5 demonstration datasets |

```bash
# Check which files are tracked by LFS
git lfs ls-files

# Pull all LFS files
git lfs pull

# Check if a file is an LFS pointer or real content
git lfs pointer --check bc_localtarget_antistall_9of50/policy/best.pt
```

---

## Notes

- Videos are **intentionally not committed**. Evaluation evidence is log-based only.
- The **anti-stall guard** is a critical component that distinguishes this working recipe
  from earlier failed attempts. Without it, the gripper stalls and the stacking fails.
- The **planner-forced discrete gripper** (binary open/close) was critical for reliable stacking.
- The **LocalTarget representation** (local frame relative to target cube) allows the policy
  to generalize across cube positions better than a global frame.
- Successful runs (main eval): 4, 5, 9, 22, 31, 33, 36, 43, 49
- Successful runs (sanity eval): 2, 3
- **Stage C (RL fine-tuning) was not completed.** Real-robot deployment requires additional work.
- Expected environment: see [`docs/environment_expected.md`](docs/environment_expected.md)
- All source file paths: see [`docs/file_sources.txt`](docs/file_sources.txt)
- Full package manifest: see [`docs/package_manifest.txt`](docs/package_manifest.txt)
