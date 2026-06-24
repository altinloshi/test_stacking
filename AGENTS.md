# test_stacking

Franka 5-cube stacking experiments in IsaacLab — Behavior Cloning and RL policies.

See `README.md` for the full reproduction guide and `commands.md` for copy-paste commands.

## Cursor Cloud specific instructions

### What this repo is
A robotics ML research repo. There are **no web servers, APIs, or databases** — "running the app" means executing a policy/training script inside **NVIDIA IsaacSim + IsaacLab** on a **CUDA RTX GPU**. There is no test suite and no lint config in the repo.

The only pip-installable, repo-local package is the Isaac Lab task extension at `source/franka_5cube_stack` (editable install). Its `install_requires` is empty by design — `isaaclab`, `rsl_rl`, and `torch` are expected to come from a host IsaacLab environment.

### GPU / IsaacSim limitation (important)
The Cursor Cloud VM is **CPU-only** (no `nvidia-smi`) and IsaacSim is not pip-installable here, so the GPU-bound entry points **cannot run** in this environment:
- `scripts/camera_once/train_reach_ppo.py` (PPO training)
- `scripts/camera_once/play_reach_ppo.py` (policy playback)
- `scripts/camera_once/smoke_camera_once.py --sim` (live sim run)
- importing `franka_5cube_stack.camera_once.env_cfg` / `mdp` (they `import isaaclab`)

These fail with `ModuleNotFoundError: No module named 'isaaclab'`, which is expected on this VM, not a setup error. To run them, use a machine with an RTX GPU + IsaacLab and launch via `./isaaclab.sh -p <script>` (see `README.md`).

### What DOES run here (GPU-free)
The import-only smoke test exercises the core functionality (Gymnasium task registration + observation/action space contract) without a GPU:
```bash
python3 scripts/camera_once/smoke_camera_once.py
```
Expected last line: `SMOKE_OK=True`. The task `Isaac-Stack-5Cube-Franka-CameraOnce-Reach-RSLRL-v0` registers via string entry points, so `import franka_5cube_stack.camera_once` only needs `gymnasium` (+ `numpy` for the smoke script).

### Dependencies / environment
- System Python 3.12 is used. Ubuntu 24.04 is PEP-668 "externally managed", so pip installs use `--break-system-packages` (no venv needed; the update script handles this).
- The `bc_localtarget_antistall_9of50/` package is **placeholders only** in this checkout — its scripts, `best.pt` checkpoint, and `*.hdf5` datasets are Git LFS assets that are not present, so it cannot be run/evaluated here. The committed eval `.log` files under `bc_localtarget_antistall_9of50/eval/.../` are the saved evidence and can be inspected directly (see `README.md` steps 4–6).
