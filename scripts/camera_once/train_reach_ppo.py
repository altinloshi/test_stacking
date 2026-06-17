"""Train the Franka camera-once reach skill with PPO (RSL-RL).

This script is self-contained and does NOT modify any IsaacLab core file.
It registers the Gymnasium task by importing the
``franka_5cube_stack.camera_once`` package before any Hydra / runner code
resolves the task name.

Usage
-----
  python scripts/camera_once/train_reach_ppo.py --num_envs 4096

Key flags
---------
  --num_envs        Number of parallel simulation environments (default 4096).
  --max_iterations  PPO update steps (default 1500).
  --checkpoint_dir  Root directory for experiment logs (default ./runs).
  --resume          Resume from the latest checkpoint in ``checkpoint_dir``.
  --headless        Run without a render window (default True).

All standard IsaacLab AppLauncher flags (--device, --livestream, etc.) are
also accepted and forwarded to the simulator.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 1 — Register the Gymnasium task BEFORE any Isaac imports.
# The import is a deliberate side-effect; the noqa suppresses F401.
# ---------------------------------------------------------------------------
import franka_5cube_stack.camera_once  # noqa: F401

# ---------------------------------------------------------------------------
# Step 2 — Set up the Isaac Sim application.
# AppLauncher must be created before any omni / isaac imports are made.
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher  # type: ignore[import]

parser = argparse.ArgumentParser(description="Train Franka camera-once reach with PPO")
parser.add_argument("--num_envs",       type=int,   default=4096)
parser.add_argument("--max_iterations", type=int,   default=1500)
parser.add_argument("--checkpoint_dir", type=str,   default="./runs")
parser.add_argument("--resume",         action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Step 3 — Post-sim imports (must come after AppLauncher).
# ---------------------------------------------------------------------------
import os
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner  # type: ignore[import]

from franka_5cube_stack.camera_once.agents.rsl_rl_ppo_cfg import (
    FrankaCameraOnceReachPPORunnerCfg,
)

# RSL-RL VecEnv wrapper — try both known import paths.
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore[import]
except ImportError:
    from isaaclab_tasks.utils.wrappers.rsl_rl import (  # type: ignore[import]
        RslRlVecEnvWrapper,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Build runner config.
    runner_cfg = FrankaCameraOnceReachPPORunnerCfg()
    runner_cfg.device          = str(args.device) if hasattr(args, "device") else "cuda:0"
    runner_cfg.max_iterations  = args.max_iterations
    runner_cfg.resume          = args.resume

    log_root = os.path.abspath(args.checkpoint_dir)
    log_dir  = os.path.join(log_root, runner_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Create environment.
    env = gym.make(
        "Isaac-Stack-5Cube-Franka-CameraOnce-Reach-RSLRL-v0",
        num_envs=args.num_envs,
        render_mode=None,
    )
    env = RslRlVecEnvWrapper(env)

    # Build and run the PPO on-policy runner.
    runner = OnPolicyRunner(
        env,
        runner_cfg.to_dict() if hasattr(runner_cfg, "to_dict") else vars(runner_cfg),
        log_dir=log_dir,
        device=runner_cfg.device,
    )

    runner.learn(
        num_learning_iterations=runner_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
