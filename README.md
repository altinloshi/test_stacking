# test_stacking

Franka 5-cube stacking experiments in IsaacLab — Behavior Cloning and RL policies.

---

## BC LocalTarget Anti-Stall Policy Reproduction

This repository includes the reproducibility package for a Franka 5-cube stacking
target-conditioned BC RNN policy. The policy uses **LocalTarget representation**,
**anti-stall guard**, **planner-forced gripper**, and a **7D action space**.
It achieved **9/50 successful rollouts** in the saved evaluation evidence.

> Full documentation: [`bc_localtarget_antistall_9of50/README_BC_9OF50.md`](bc_localtarget_antistall_9of50/README_BC_9OF50.md)

### Policy Description

This is a **Stage A/B prototype** (Stage C RL fine-tuning was not completed):

- **Stage A:** HDF5 demonstration data from scripted StackPlanner / expert controller
- **Stage B:** Target-conditioned BC RNN policy — LocalTarget representation, anti-stall guard, planner-forced gripper
- **Stage C:** RL fine-tuning with domain randomization — **not completed**

### Results Summary

**Main evaluation — 50 runs (2026-06-19 17:55:28):**

| Metric | Value |
|--------|-------|
| Total runs | 50 |
| Successful runs | **9** |
| Success rate | **18%** |
| Failed runs | 41 |
| Successful run IDs | 4, 5, 9, 22, 31, 33, 36, 43, 49 |

**Sanity evaluation — 10 runs (2026-06-19 16:29:50):**

| Metric | Value |
|--------|-------|
| Total runs | 10 |
| Successful runs | **2** |
| Success rate | **20%** |
| Successful run IDs | 2, 3 |

### Package Structure

```
bc_localtarget_antistall_9of50/
├── policy/
│   └── best.pt                  ← RNN BC policy checkpoint (Git LFS)
├── datasets/
│   └── *.hdf5                   ← Training demonstration datasets (Git LFS)
├── scripts/
│   ├── play_bc_rnn_target_conditioned.py
│   ├── train_bc_rnn_target_conditioned.py
│   ├── run_5cube_camera_hierarchical_success_fixed.py
│   └── (localtarget, antistall, hdf5 utility scripts)
├── eval/
│   ├── localtarget_antistall_50x_eval_20260619_175528/   ← 50-run eval logs
│   ├── localtarget_antistall_10x_eval_20260619_162950/   ← 10-run sanity logs
│   ├── eval_LOCALTARGET_ANTISTALL_50runs.sh
│   └── eval_LOCALTARGET_ANTISTALL_10runs.sh
├── logs/
│   └── *.log
└── docs/
    ├── package_manifest.txt
    ├── environment_expected.md
    └── file_sources.txt
```

---

## Step 1 — Clone and Pull Large Files

```bash
git clone https://github.com/altinloshi/test_stacking.git
cd test_stacking

git lfs install
git lfs pull
```

---

## Step 2 — Setup Environment

IsaacLab must be installed separately and its Python environment activated.
See [`bc_localtarget_antistall_9of50/docs/environment_expected.md`](bc_localtarget_antistall_9of50/docs/environment_expected.md)
for the full expected environment.

```bash
# Inside the IsaacLab Python environment:

# Register the Franka 5-cube task extension
pip install -e source/franka_5cube_stack

# Install robomimic (required for BC training)
pip install robomimic

# Verify task registration
python -c "import gymnasium as gym; import franka_5cube_stack; print(gym.spec('Isaac-Stack-5Cube-Franka-IK-Rel-v0'))"
```

---

## Step 3 — Verify Saved Evaluation Evidence

No IsaacLab required for this step — just check the committed log files.

```bash
# Count successes in main 50-run eval (expected: 9)
grep -R "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log \
  | wc -l

# Count failures (expected: 41)
grep -R "FINAL PHYSICAL STACK SUCCESS=False" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log \
  | wc -l

# List successful run IDs
grep -Rl "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log \
  | sort

# Inspect a successful run (e.g., run 4)
grep -n "FINAL PHYSICAL STACK SUCCESS\|TARGET BC STEP\|guard=\|xy_gap\|z_gap" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_4.log \
  | tail -120

# Count successes in sanity 10-run eval (expected: 2)
grep -R "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_10x_eval_20260619_162950/run_*.log \
  | wc -l
```

---

## Step 4 — Run Fresh Evaluation (Requires IsaacLab)

Prerequisites: IsaacLab env active, franka_5cube_stack installed, `best.pt` present.

```bash
# Run 50-episode evaluation
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_50runs.sh

# Run 10-episode sanity evaluation
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_10runs.sh

# Run a single episode directly
python bc_localtarget_antistall_9of50/scripts/play_bc_rnn_target_conditioned.py \
    --task Isaac-Stack-5Cube-Franka-IK-Rel-v0 \
    --checkpoint bc_localtarget_antistall_9of50/policy/best.pt \
    --run_id 0 \
    --num_envs 1 \
    --headless \
    --output_dir /tmp/bc_test_run/
```

---

## Step 5 — Train from Scratch (Optional)

Only needed if you want to reproduce the full pipeline. Do not retrain to verify results.

```bash
# Verify datasets exist
ls bc_localtarget_antistall_9of50/datasets/*.hdf5

# Train BC RNN policy (60 demos, 20 epochs — matching the saved checkpoint)
python bc_localtarget_antistall_9of50/scripts/train_bc_rnn_target_conditioned.py \
    --dataset bc_localtarget_antistall_9of50/datasets/ \
    --num_demos 60 \
    --num_epochs 20 \
    --output_dir logs/bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch/ \
    2>&1 | tee /tmp/train_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch.log

# Copy the best checkpoint into the package
cp logs/bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch/best.pt \
   bc_localtarget_antistall_9of50/policy/best.pt
```

---

## Step 6 — Collect Demonstrations (Optional)

Only needed if reproducing from scratch. The original training used 60 expert demos.

```bash
# Collect via scripted expert (StackPlanner)
python bc_localtarget_antistall_9of50/scripts/run_5cube_camera_hierarchical_success_fixed.py \
    --task Isaac-Stack-5Cube-Franka-IK-Rel-v0 \
    --num_envs 50 \
    --num_steps 700 \
    --output_dir bc_localtarget_antistall_9of50/datasets/ \
    --headless \
    2>&1 | tee /tmp/collect_TARGET_CONDITIONED_BC_50env_700steps.log

# Verify collected HDF5 files
python -c "
import h5py, sys
for f in sys.argv[1:]:
    with h5py.File(f, 'r') as h:
        demos = list(h.get('data', h).keys())
        print(f'{f}: {len(demos)} demos')
" bc_localtarget_antistall_9of50/datasets/*.hdf5
```

---

## Step 7 — Add Files from Local Machine

If you're the original developer, add the binary/log files from your local machine:

```bash
# Add policy checkpoint (Git LFS)
cp /home/a0loshi1/robotics/IsaacLab/logs/bc_detector_expert/bc_rnn_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch/best.pt \
   bc_localtarget_antistall_9of50/policy/best.pt
git add bc_localtarget_antistall_9of50/policy/best.pt
git commit -m "Add BC policy checkpoint best.pt via LFS"
git push

# Add main 50-run eval logs
cp -r /tmp/localtarget_antistall_50x_eval_20260619_175528/ \
   bc_localtarget_antistall_9of50/eval/
git add bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/
git commit -m "Add main 50-run evaluation logs (9/50 successes)"
git push

# Add sanity 10-run eval logs
cp -r /tmp/localtarget_antistall_10x_eval_20260619_162950/ \
   bc_localtarget_antistall_9of50/eval/
git add bc_localtarget_antistall_9of50/eval/localtarget_antistall_10x_eval_20260619_162950/
git commit -m "Add sanity 10-run evaluation logs (2/10 successes)"
git push

# Add HDF5 datasets (Git LFS)
cp /home/a0loshi1/robotics/IsaacLab/datasets/*TARGET_CONDITIONED*.hdf5 \
   bc_localtarget_antistall_9of50/datasets/ 2>/dev/null || true
cp /home/a0loshi1/robotics/IsaacLab/datasets/*LOCALTARGET*.hdf5 \
   bc_localtarget_antistall_9of50/datasets/ 2>/dev/null || true
git add bc_localtarget_antistall_9of50/datasets/*.hdf5
git commit -m "Add BC training datasets via LFS"
git push

# Add scripts
for f in play_bc_rnn_target_conditioned.py train_bc_rnn_target_conditioned.py \
          run_5cube_camera_hierarchical_success_fixed.py; do
    cp /home/a0loshi1/robotics/IsaacLab/scripts/custom/$f \
       bc_localtarget_antistall_9of50/scripts/ 2>/dev/null || true
done
git add bc_localtarget_antistall_9of50/scripts/
git commit -m "Add BC policy scripts"
git push

# Add optional training logs
for f in /tmp/train_TARGET_CONDITIONED_LOCALTARGET_60demos_20epoch.log \
         /tmp/play_TARGET_CONDITIONED_LOCALTARGET_700_antistall.log \
         /tmp/collect_TARGET_CONDITIONED_BC_50env_700steps.log; do
    [ -f "$f" ] && cp "$f" bc_localtarget_antistall_9of50/logs/
done
git add bc_localtarget_antistall_9of50/logs/ 2>/dev/null || true
git diff --cached --quiet || git commit -m "Add training and collection logs"
git push
```

---

## Git LFS Configuration

Large binary files are tracked via Git LFS (`.gitattributes`):

| Pattern | Type |
|---------|------|
| `*.pt`  | PyTorch model checkpoints |
| `*.pth` | PyTorch state dicts |
| `*.hdf5`| HDF5 demonstration datasets |

```bash
# Verify LFS-tracked files
git lfs ls-files

# Check a specific file is real content (not a pointer)
git lfs pointer --check bc_localtarget_antistall_9of50/policy/best.pt
```

---

See [`bc_localtarget_antistall_9of50/README_BC_9OF50.md`](bc_localtarget_antistall_9of50/README_BC_9OF50.md)
for the full documentation including environment setup, training details, and log inspection commands.

---
