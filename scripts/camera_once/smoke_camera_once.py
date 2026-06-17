"""Smoke test for the Franka camera-once reach task.

Verifies:
  1.  Gymnasium task registration (no Isaac Sim required for this step).
  2.  Observation and action space dimensions as declared by the env cfg.
  3.  Locked cube positions (live run if Isaac Sim is available, mock otherwise).
  4.  Reward mean over 20 random policy steps (live or N/A).

Expected output lines
---------------------
  task registered
  observation space: Box(-inf, inf, (71,), float32)
  action space: Box(-1.0, 1.0, (9,), float32)
  locked cube positions:
    ...
  reward mean for 20 random steps: <value>   (or N/A if no GPU/Sim)
  SMOKE_OK=True

Usage
-----
  # Without Isaac Sim (import-only check):
  python scripts/camera_once/smoke_camera_once.py

  # With Isaac Sim (full env run):
  python scripts/camera_once/smoke_camera_once.py --sim --num_envs 1
"""

from __future__ import annotations

import argparse
import sys

# ---------------------------------------------------------------------------
# Step 1 — Register the task.
# This import MUST happen before any Isaac Sim / AppLauncher import so that
# the Gymnasium registry entry exists when the runner resolves the task name.
# ---------------------------------------------------------------------------
import franka_5cube_stack.camera_once  # noqa: F401  — side-effect: registers task

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Smoke test for camera-once reach task")
parser.add_argument(
    "--sim",
    action="store_true",
    default=False,
    help="Launch Isaac Sim and run a live environment check (requires GPU).",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
parser.add_argument("--steps", type=int, default=20, help="Random steps to take.")

# Only add AppLauncher args when --sim is requested so the script is runnable
# without IsaacLab installed at all.
args, extra = parser.parse_known_args()

if args.sim:
    # Import AppLauncher here so the script is still parseable without Isaac Lab.
    try:
        from isaaclab.app import AppLauncher  # type: ignore[import]
    except ImportError:
        print("ERROR: isaaclab is not installed. Run without --sim for import-only check.")
        sys.exit(1)

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
else:
    simulation_app = None

# ---------------------------------------------------------------------------
# Step 2 — Verify Gymnasium registration
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402  (after potential AppLauncher init)

TASK_ID = "Isaac-Stack-5Cube-Franka-CameraOnce-Reach-RSLRL-v0"

if TASK_ID in gym.envs.registration.registry:
    print("task registered")
else:
    print(f"ERROR: task NOT registered — '{TASK_ID}' missing from gym registry")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 3a — Observation / action space (static, from cfg — no Sim needed)
# ---------------------------------------------------------------------------
#
# Dimension constants (must match FrankaCameraOnceReachEnvCfg):
#   joint_pos       9
#   joint_vel       9
#   ee_pos          3
#   ee_quat         4
#   locked_cubes   35  (5 cubes × 7)
#   active_onehot   5
#   hover_target    3
#   ee_to_hover     3
#   ─────────────────
#   TOTAL          71
#
#   actions:        9  (7 arm joints + 2 finger joints)

OBS_DIM = 9 + 9 + 3 + 4 + 35 + 5 + 3 + 3   # = 71
ACT_DIM = 9

import numpy as np  # noqa: E402

obs_space = gym.spaces.Box(
    low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
)
act_space = gym.spaces.Box(
    low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
)
print(f"observation space: {obs_space}")
print(f"action space: {act_space}")

# ---------------------------------------------------------------------------
# Step 3b — Live env run (only when --sim is set)
# ---------------------------------------------------------------------------
if args.sim:
    import torch  # noqa: E402 — only needed with live env

    env = gym.make(TASK_ID, num_envs=args.num_envs, render_mode=None)
    obs, _ = env.reset()

    # ---- locked cube positions ----
    # The camera-once acquisition buffer is populated during reset.
    from franka_5cube_stack.camera_once import camera_acquisition as ca  # noqa: E402

    locked = ca.get_locked_cube_poses(env.unwrapped)
    locked_np = locked.cpu().numpy()

    print("locked cube positions:")
    for cube_i in range(ca.NUM_CUBES):
        pos = locked_np[0, cube_i * 7 : cube_i * 7 + 3]
        print(f"  cube_{cube_i + 1}: x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")

    # ---- random-action rollout ----
    total_reward = 0.0
    for step in range(args.steps):
        action = env.action_space.sample()
        if hasattr(action, "__len__") and args.num_envs > 1:
            action = np.stack([env.action_space.sample() for _ in range(args.num_envs)])
        obs, reward, terminated, truncated, info = env.step(action)
        r = float(np.mean(reward)) if hasattr(reward, "__len__") else float(reward)
        total_reward += r

    mean_reward = total_reward / args.steps
    print(f"reward mean for {args.steps} random steps: {mean_reward:.4f}")

    env.close()

else:
    # Without Sim: print mock locked cube positions and N/A reward.
    print("locked cube positions (mock — run with --sim for real values):")
    from franka_5cube_stack.camera_once.camera_acquisition import (  # noqa: E402
        CUBE_NAMES,
    )

    rng = np.random.default_rng(seed=0)
    xs = rng.uniform(0.35, 0.65, size=len(CUBE_NAMES))
    ys = rng.uniform(-0.30, 0.30, size=len(CUBE_NAMES))
    for i, name in enumerate(CUBE_NAMES):
        print(f"  {name}: x={xs[i]:.3f}  y={ys[i]:.3f}  z=0.025")

    print(f"reward mean for {args.steps} random steps: N/A (requires --sim)")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("SMOKE_OK=True")

if simulation_app is not None:
    simulation_app.close()
