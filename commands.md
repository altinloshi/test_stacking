# test_stacking

Franka 5-cube stacking experiments in IsaacLab.

This repository currently contains a **Behavior Cloning (BC) policy package** for the Franka 5-cube stacking task. RL fine-tuning is **not included yet**.

---

## Current status

So far, expert demonstrations were generated in IsaacLab and used to train a **target-conditioned BC RNN policy**. The policy uses:

* LocalTarget task context
* 7D action output: `dx, dy, dz, rx, ry, rz, gripper`
* anti-stall guard during rollout
* forced gripper logic

The current best evaluation result is:

```text
9 successful rollouts out of 50
Success rate: 18%
```

Successful run IDs:

```text
4, 5, 9, 22, 31, 33, 36, 43, 49
```

The camera/RGB-D part is used to provide visual observations of the cube scene, so the project can move toward estimating cube positions from vision instead of relying only on privileged simulator state.

---

## Policy package

The BC policy package is stored here:

```text
bc_localtarget_antistall_9of50/
```

Main files:

```text
bc_localtarget_antistall_9of50/
├── policy/
│   └── best.pt
├── datasets/
│   └── *.hdf5
├── scripts/
│   └── custom scripts for training and playing the policy
├── eval/
│   └── saved evaluation logs
├── logs/
│   └── training/play logs
└── docs/
    └── environment and file notes
```

The main checkpoint is:

```text
bc_localtarget_antistall_9of50/policy/best.pt
```

The 50-run evaluation logs are:

```text
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/
```

---

## Setup after cloning

This repository uses Git LFS for large files such as `.pt` and `.hdf5`.

```bash
git clone https://github.com/altinloshi/test_stacking.git
cd test_stacking

git lfs install
git lfs pull
```

Check that the policy checkpoint was downloaded correctly:

```bash
ls -lh bc_localtarget_antistall_9of50/policy/best.pt
head -5 bc_localtarget_antistall_9of50/policy/best.pt
```

If the output starts with:

```text
version https://git-lfs.github.com/spec/v1
```

then the real checkpoint was not downloaded. Run:

```bash
git lfs pull
```

---

## Verify saved evaluation result

Count successful rollouts:

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

Expected result:

```text
9
```

Count failed rollouts:

```bash
grep -R "FINAL PHYSICAL STACK SUCCESS=False" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | wc -l
```

Expected result:

```text
41
```

List successful runs:

```bash
grep -Rl "FINAL PHYSICAL STACK SUCCESS=True" \
  bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/run_*.log | sort
```

---

## Run the policy in IsaacLab

A working IsaacLab / IsaacSim installation is required.

First install the custom task package from this repository:

```bash
cd /path/to/test_stacking
python -m pip install -e source/franka_5cube_stack
```

Then run the BC policy from the IsaacLab folder:

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

Replace:

```text
/path/to/test_stacking
/path/to/IsaacLab
```

with the actual paths on your machine.

---

## Important note

The saved `.log` files are evaluation evidence, not video replays. To see the robot move, the policy must be run again inside IsaacLab.

The result may differ on another machine because IsaacSim, IsaacLab, GPU driver, CUDA, PyTorch, and random seeds can affect simulation behavior.

---

## Troubleshooting

If `best.pt` is very small or shows Git LFS pointer text:

```bash
git lfs pull
```

If the task is not found:

```bash
cd /path/to/test_stacking
python -m pip install -e source/franka_5cube_stack
```

If there are import errors, check the IsaacLab, IsaacSim, PyTorch, CUDA, and robomimic versions.

If the GUI does not open, make sure you are not using `--headless` and that IsaacSim GUI works on your machine.

---

## Earlier experiment

The folder below contains an earlier camera-once PPO/reach experiment:

```text
scripts/camera_once/
source/franka_5cube_stack/franka_5cube_stack/camera_once/
```
