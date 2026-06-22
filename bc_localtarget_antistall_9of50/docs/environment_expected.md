# Expected Environment

This package was developed and evaluated on a local IsaacLab / IsaacSim setup.

Fill in exact values from the original machine if available. The placeholders below reflect what is known.

---

## Hardware

| Component | Value |
|-----------|-------|
| OS | *(fill in — e.g. Ubuntu 22.04)* |
| GPU | *(fill in — e.g. NVIDIA RTX 4090)* |
| NVIDIA driver | *(fill in — e.g. 535.x)* |
| CUDA | *(fill in — e.g. 12.2)* |
| Device used at eval | `cuda:0` |

---

## Software

| Package | Value |
|---------|-------|
| IsaacSim version | *(fill in — e.g. 4.2.0)* |
| IsaacLab commit / version | *(fill in — e.g. v1.x or commit hash)* |
| Python version | *(fill in — e.g. 3.10.x)* |
| PyTorch version | *(fill in — e.g. 2.3.x+cu121)* |
| torchvision version | *(fill in)* |
| robomimic version | *(fill in — e.g. 0.3.x)* |
| h5py version | *(fill in)* |
| numpy version | *(fill in)* |

---

## Custom packages (installed editable)

- `source/franka_5cube_stack` — registers `Isaac-Stack-5Cube-Franka-IK-Rel-v0`

Install with:

```bash
cd /path/to/test_stacking
python -m pip install -e source/franka_5cube_stack
```

---

## Important notes on reproducibility

> Exact results may differ across IsaacSim, IsaacLab, GPU, driver, PyTorch, and CUDA versions.

Known sources of variation:

- IsaacSim physics simulation is deterministic given identical hardware, driver, and CUDA version,
  but results can differ across GPU models and driver updates.
- CUDA random seed behaviour can differ across PyTorch versions.
- The BC policy (18% success rate on 50 runs) is brittle by design — Stage C RL fine-tuning
  was not completed. Small environment changes can noticeably shift success rate.
- The anti-stall guard and planner-forced gripper are coded in the custom play script.
  Do not use the standard IsaacLab play script without these components.

---

## Reproducing the original evaluation

The original 9/50 evaluation logs are in:

```
bc_localtarget_antistall_9of50/eval/localtarget_antistall_50x_eval_20260619_175528/
```

These logs are the ground truth. To reproduce fresh runs:

```bash
bash bc_localtarget_antistall_9of50/eval/eval_LOCALTARGET_ANTISTALL_50runs.sh
```

See `README_BC_9OF50.md` Section 13 for full details.
