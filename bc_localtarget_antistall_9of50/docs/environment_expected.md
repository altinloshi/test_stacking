# Expected Environment

This package was developed around a local IsaacLab setup.
The following describes the environment used to collect demonstrations, train the policy,
and run the saved evaluations.

---

## Platform

| Component         | Value                                      |
|-------------------|--------------------------------------------|
| OS                | Ubuntu 22.04                               |
| GPU               | NVIDIA RTX (local workstation)             |
| NVIDIA driver     | 535.x or newer (CUDA 12.x compatible)      |
| CUDA              | 12.1 – 12.4                                |
| Device used       | `cuda:0`                                   |

---

## Software Stack

| Package           | Version / Notes                                        |
|-------------------|--------------------------------------------------------|
| IsaacSim          | 4.2.0 (bundled with IsaacLab)                          |
| IsaacLab          | 1.x (cloned from isaac-sim/IsaacLab, main branch)      |
| Python            | 3.10 (provided by IsaacLab conda/venv)                 |
| PyTorch           | 2.2.x (with CUDA 12.x backend)                         |
| robomimic         | 0.3.x (installed via pip into IsaacLab env)            |
| franka_5cube_stack| 0.1.0 (this repo's extension, installed via pip)        |

---

## Task Registration

The Isaac Lab task used for all evaluations:

```
Isaac-Stack-5Cube-Franka-IK-Rel-v0
```

This task is registered by the `franka_5cube_stack` extension in this repo.
It must be installed before any script is run:

```bash
# From repo root (test_stacking/)
pip install -e source/franka_5cube_stack
```

---

## Notes on Reproducibility

Exact reproduction may differ across:
- IsaacSim versions (physics engine changes between releases)
- IsaacLab versions (API changes, environment changes)
- GPU model and CUDA version (floating-point non-determinism)
- PyTorch version (numerics, RNG)
- NVIDIA driver version (physics engine behavior)

The 9/50 success rate was measured on the specific environment above.
Results on a different machine or version may vary.

---

## Stage A/B Policy — Not Full Sim-to-Real

This policy is a **Stage A/B prototype**, not a full sim-to-real policy:

- **Stage A**: HDF5 demonstration data collected from scripted StackPlanner / expert controller
- **Stage B**: Target-conditioned BC RNN policy with LocalTarget representation, anti-stall guard, and planner-forced gripper
- **Stage C**: RL fine-tuning with domain randomization — **not completed**

Do not expect real-robot deployment without Stage C fine-tuning.
