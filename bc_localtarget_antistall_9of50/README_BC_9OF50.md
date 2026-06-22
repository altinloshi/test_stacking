# BC LocalTarget Anti-Stall Policy — Franka 5-Cube Stacking

---

## 1. Overview

This package contains a reproducibility package for a Franka 5-cube stacking Behavior Cloning policy trained and evaluated inside IsaacLab / IsaacSim.

It includes:

- Trained policy checkpoint (`policy/best.pt`, via Git LFS)
- Relevant HDF5 demonstration datasets (`datasets/*.hdf5`, via Git LFS)
- Custom train / play / eval / source scripts (`scripts/`)
- 9/50 main evaluation logs (`eval/localtarget_antistall_50x_eval_20260619_175528/`)
- 2/10 sanity evaluation logs (`eval/localtarget_antistall_10x_eval_20260619_162950/`)
- Training / play / collection logs (`logs/`)
- Documentation (`docs/`)

The goal is to let another engineer clone the repository, pull the trained policy and dataset files, inspect the evaluation evidence, and rerun the policy inside a matching IsaacLab environment.

---

## 2. What Was Actually Done

**This is not a full sim-to-real RL fine-tuned policy.** It is a Stage A / Stage B prototype:

```
Stage A  DATA:
  Scripted expert / StackPlanner records RGB-D / camera data,
  proprioception, 7D expert actions, and cube/target information
  into HDF5 demonstration files.

Stage B  BC:
  Target-conditioned BC RNN trained on Stage A data.
  Uses LocalTarget representation, anti-stall guard,
  and planner-forced discrete gripper.
  Output: working but brittle policy (18% success rate).

Stage C  RL:
  RL fine-tuning with domain randomization.
  NOT completed.
```

### Stage A — Data Collection

- Scripted StackPlanner / expert controller runs 5-cube stacking in IsaacLab
- Observations: RGB-D camera, proprioception, cube poses, target index
- Actions: 7D delta EE pose + gripper (`dx, dy, dz, rx, ry, rz, gripper`)
- Output: HDF5 demonstration files

### Stage B — Behavior Cloning

Policy type:

- Target-conditioned Behavior Cloning
- LocalTarget representation (local coordinate frame relative to current target cube)
- RNN policy checkpoint (temporal dependencies across stacking steps)
- Anti-stall / guard layer (detects and recovers from gripper stall)
- Planner-forced discrete gripper (binary open / close via planner override)
- 7D action space: `dx, dy, dz, rx, ry, rz, gripper`

### Stage C — Not Done

RL fine-tuning with domain randomization was not completed. The policy generalises weakly (18%) because it was trained only on scripted-expert demonstrations without environment variation or online RL correction.

---

## 3. Result Summary

### Main Evaluation — 50 runs (2026-06-19 17:55:28)

| Metric | Value |
|--------|-------|
| Total runs | 50 |
| Successful runs | **9** |
| Failed runs | 41 |
| Success rate | **18%** |
| Successful run IDs | 4, 5, 9, 22, 31, 33, 36, 43, 49 |

### Sanity Evaluation — 10 runs (2026-06-19 16:29:50)

| Metric | Value |
|--------|-------|
| Total runs | 10 |
| Successful runs | **2** |
| Failed runs | 8 |
| Success rate | **20%** |
| Successful run IDs | 2, 3 |

> Videos are intentionally not committed to this repository.
> Evaluation evidence is preserved as `.log` files only.

---

## 4. Package Structure

```
bc_localtarget_antistall_9of50/
├── policy/
│   └── best.pt                    ← Trained RNN BC checkpoint (Git LFS)
├── datasets/
│   └── *.hdf5                     ← Demonstration datasets (Git LFS)
├── scripts/
│   └── custom/
│       ├── bc_detector_expert/
│       │   ├── play_bc_rnn_target_conditioned.py
│       │   └── train_bc_rnn_target_conditioned.py
│       └── run_5cube_camera_hierarchical_success_fixed.py
├── eval/
│   ├── localtarget_antistall_50x_eval_20260619_175528/
│   │   └── run_0.log … run_49.log ← Main 50-run evaluation logs
│   ├── localtarget_antistall_10x_eval_20260619_162950/
│   │   └── run_0.log … run_9.log  ← Sanity 10-run evaluation logs
│   ├── eval_LOCALTARGET_ANTISTALL_50runs.sh
│   └── eval_LOCALTARGET_ANTISTALL_10runs.sh
├── logs/
│   └── *.log                      ← Training / play / collection logs
├── docs/
│   ├── package_manifest.txt
│   ├── environment_expected.md
│   └── file_sources.txt
└── README_BC_9OF50.md             ← This file
```

---

## 5. Clone and Pull Large Files

```bash
git clone https://github.com/altinloshi/test_stacking.git
cd test_stacking
git lfs install
git lfs pull
```

> **Important:** If `best.pt` is only 134 bytes, it is a Git LFS pointer — the real model was not downloaded.
> Run `git lfs pull` to fetch the actual checkpoint.

---

## 6. Verify Checkpoint

```bash
ls -lh bc_localtarget_antistall_9of50/policy/best.pt
head -5 bc_localtarget_antistall_9of50/policy/best.pt
```

If `head` prints something like:

```
version https://git-lfs.github.com/spec/v1
oid sha256:...
size ...
```

then the checkpoint is still a pointer. Run:

```bash
git lfs pull
```

A valid PyTorch checkpoint will not be human-readable ASCII.

---

## 7. Verify Datasets

```bash
find bc_localtarget_antistall_9of50/datasets -type f \( -name "*.hdf5" -o -name "*.h5" \) \
  -exec ls -lh {} \;
```

Each `.hdf5` file should be several MB or larger (not 134 bytes).

---

## 8. Verify 9/50 Evidence

Count successful runs in the main 50-run evaluation:

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

Expected output:

```
9
```

Count failed runs:

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=False" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

Expected output:

```
41
```

---

## 9. List Successful Runs

```bash
grep -Rl "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | sort
```

Expected output:

```
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_4.log
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_5.log
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_9.log
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_22.log
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_31.log
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_33.log
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_36.log
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_43.log
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_49.log
```

---

## 10. Inspect One Successful Log

Inspect key events in run 4:

```bash
grep -n "FINAL PHYSICAL STACK SUCCESS\|TARGET BC STEP\|guard=\|xy_gap\|z_gap" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_4.log | tail -120
```

---

## 11. Run the Policy Visually in IsaacLab

A working IsaacLab / IsaacSim installation is required. The eval logs are not simulator videos. To see the robot move, rerun the policy live.

```bash
cd /path/to/IsaacLab

./isaaclab.sh -p /path/to/test_stacking/bc_localtarget_antistall_9of50/scripts/custom/bc_detector_expert/play_bc_rnn_target_conditioned.py \
  --base_script /path/to/test_stacking/bc_localtarget_antistall_9of50/scripts/custom/run_5cube_camera_hierarchical_success_fixed.py \
  --checkpoint /path/to/test_stacking/bc_localtarget_antistall_9of50/policy/best.pt \
  --task Isaac-Stack-5Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --max_steps 700 \
  --enable_cameras \
  --disable_yaw_grasp \
  --device cuda:0
```

Replace `/path/to/IsaacLab` and `/path/to/test_stacking` with your actual paths.

---

## 12. Install Custom Task (if Task Not Found)

If you see `KeyError: Isaac-Stack-5Cube-Franka-IK-Rel-v0`, install the custom task package:

```bash
cd /path/to/test_stacking
python -m pip install -e source/franka_5cube_stack
```

Then rerun the play command.

---

## 13. Headless Batch Evaluation

```bash
cd /path/to/test_stacking
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_50runs.sh
```

After it finishes, check the result:

```bash
LATEST=$(ls -td /tmp/localtarget_antistall_50x_eval_* | head -1)
echo "$LATEST"
grep -R "FINAL PHYSICAL STACK SUCCESS=True"  "$LATEST"/run_*.log | wc -l
grep -R "FINAL PHYSICAL STACK SUCCESS=False" "$LATEST"/run_*.log | wc -l
grep -Rl "FINAL PHYSICAL STACK SUCCESS=True" "$LATEST"/run_*.log | sort
```

---

## 14. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `best.pt` is 134 bytes / ASCII | Run `git lfs pull` to fetch the real checkpoint |
| `KeyError: Isaac-Stack-5Cube-Franka-IK-Rel-v0` | Install `source/franka_5cube_stack` with `pip install -e` |
| Import errors for custom modules | Ensure IsaacLab, IsaacSim, PyTorch, robomimic, and Python versions match `docs/environment_expected.md` |
| GUI not opening | Remove `--headless`; verify IsaacSim GUI is working on your machine |
| Success rate differs from 9/50 | Simulation can vary across Isaac versions, GPU, driver, CUDA, and random seeds |
| Eval logs are empty / wrong paths | Confirm log files are present — run `find bc_localtarget_antistall_9of50/eval -name "*.log" | wc -l` |
| Eval logs are not visual replays | Correct — the `.log` files are text output only. To see the robot, rerun the policy (Section 11) |

---

## 15. Git LFS Setup

Large binary files are tracked via Git LFS. To set up LFS tracking on a new machine:

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.hdf5"
git add .gitattributes
```

> **Do not track videos.** Videos are intentionally not committed to this repository.

Verify which files are stored in LFS:

```bash
git lfs ls-files
```

Expected entries:

```
<hash> * bc_localtarget_antistall_9of50/policy/best.pt
<hash> * bc_localtarget_antistall_9of50/datasets/<name>.hdf5
...
```

---

## Notes

- The anti-stall guard is the key component that distinguishes this working recipe from earlier failed attempts without it.
- The planner-forced discrete gripper (binary open / close) was critical for reliable stacking.
- Stage C (RL fine-tuning with domain randomisation) was not completed; it would likely push success rate significantly higher.
- Success rate is expected to vary across IsaacSim / IsaacLab versions, GPU hardware, driver versions, CUDA, and random seeds.
