# test_stacking

Franka 5-cube stacking experiments in IsaacLab — Behavior Cloning and RL policies.

---

## BC LocalTarget Anti-Stall Policy Reproduction

This repository includes the reproducibility package for a Franka 5-cube stacking target-conditioned BC RNN policy.
The policy uses LocalTarget representation, anti-stall guard, planner-forced gripper, and a 7D action space.
It achieved **9/50 successful rollouts** in the saved evaluation evidence.

Full package documentation: [`bc_localtarget_antistall_9of50/README_BC_9OF50.md`](bc_localtarget_antistall_9of50/README_BC_9OF50.md)

### Result summary

| Evaluation | Runs | Successes | Rate | Successful run IDs |
|------------|------|-----------|------|-------------------|
| Main (2026-06-19 17:55:28) | 50 | **9** | **18%** | 4, 5, 9, 22, 31, 33, 36, 43, 49 |
| Sanity (2026-06-19 16:29:50) | 10 | **2** | **20%** | 2, 3 |

> **Stage A/B prototype only.** Stage C RL fine-tuning was not completed.
> Videos are intentionally not committed. Evidence is `.log` files only.
> Git LFS is required to pull `best.pt` and `*.hdf5` files.

### Package structure

```
bc_localtarget_antistall_9of50/
├── policy/
│   └── best.pt                                   ← BC RNN checkpoint (Git LFS)
├── datasets/
│   └── *.hdf5                                    ← Training demos (Git LFS)
├── scripts/
│   └── custom/
│       ├── bc_detector_expert/
│       │   ├── play_bc_rnn_target_conditioned.py
│       │   └── train_bc_rnn_target_conditioned.py
│       └── run_5cube_camera_hierarchical_success_fixed.py
├── eval/
│   ├── localtarget_antistall_50x_eval_20260619_175528/  ← 50-run evidence logs
│   ├── localtarget_antistall_10x_eval_20260619_162950/  ← 10-run sanity logs
│   ├── eval_LOCALTARGET_ANTISTALL_50runs.sh
│   └── eval_LOCALTARGET_ANTISTALL_10runs.sh
├── logs/                                         ← training/play/collection logs
└── docs/
    ├── package_manifest.txt
    ├── environment_expected.md
    └── file_sources.txt
```

---

### Step 1 — Clone and pull large files

```bash
git clone https://github.com/altinloshi/test_stacking.git
cd test_stacking
git lfs install
git lfs pull
```

> If `best.pt` is 134 bytes after cloning, it is a Git LFS pointer.
> Run `git lfs pull` to download the real checkpoint.

---

### Step 2 — Verify checkpoint

```bash
ls -lh bc_localtarget_antistall_9of50/policy/best.pt
head -5 bc_localtarget_antistall_9of50/policy/best.pt
```

If `head` prints lines beginning with `version https://git-lfs.github.com/spec/v1`, run:

```bash
git lfs pull
```

---

### Step 3 — Verify datasets

```bash
find bc_localtarget_antistall_9of50/datasets -type f \( -name "*.hdf5" -o -name "*.h5" \) \
  -exec ls -lh {} \;
```

Each `.hdf5` file should be several MB or larger, not 134 bytes.

---

### Step 4 — Verify 9/50 evidence

Count successes:

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

Expected output: `9`

Count failures:

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=False" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

Expected output: `41`

---

### Step 5 — List successful runs

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

### Step 6 — Inspect a successful log

```bash
grep -n "FINAL PHYSICAL STACK SUCCESS\|TARGET BC STEP\|guard=\|xy_gap\|z_gap" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_4.log | tail -120
```

---

### Step 7 — Install custom task (if task not found)

```bash
cd /path/to/test_stacking
python -m pip install -e source/franka_5cube_stack
```

Required if you see `KeyError: Isaac-Stack-5Cube-Franka-IK-Rel-v0`.

---

### Step 8 — Run the policy visually in IsaacLab

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

> The eval logs are not video replays. To see the robot move, run the policy live as shown above.

---

### Step 9 — Headless batch evaluation

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

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `best.pt` is 134 bytes or ASCII text | Run `git lfs pull` |
| `KeyError: Isaac-Stack-5Cube-Franka-IK-Rel-v0` | `pip install -e source/franka_5cube_stack` |
| Import errors for custom modules | Check IsaacLab/IsaacSim/PyTorch/robomimic/CUDA/driver versions against `docs/environment_expected.md` |
| GUI not opening | Remove `--headless`; verify IsaacSim GUI works on your machine |
| Success rate differs from 9/50 | Expected — varies across Isaac versions, GPU, driver, CUDA, seeds |
| No `run_*.log` files found | Add eval logs from local machine (see `eval/.../PLACE_RUN_LOGS_HERE.txt`) |

---

### Git LFS setup (for new contributors)

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.hdf5"
git add .gitattributes
```

Do not track videos. Videos are intentionally not committed to this repository.

Verify LFS-tracked files:

```bash
git lfs ls-files
```

---

## Other experiments

### camera_once PPO reach (earlier experiment)

Scripts: `scripts/camera_once/`
Source: `source/franka_5cube_stack/franka_5cube_stack/camera_once/`
