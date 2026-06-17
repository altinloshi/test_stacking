"""Play / evaluate a trained Franka camera-once reach policy.

Loads a checkpoint produced by ``train_reach_ppo.py`` and runs the policy in
inference mode.  The Isaac Sim viewport is opened by default so you can watch
the robot reach for the hover target above the active cube.

Usage
-----
  python scripts/camera_once/play_reach_ppo.py \\
      --checkpoint ./runs/franka_camera_once_reach/<run>/model_1500.pt \\
      --num_envs 16

Key flags
---------
  --checkpoint   Path to a ``.pt`` checkpoint file (required).
  --num_envs     Number of parallel evaluation environments (default 16).
  --video        Record an mp4 video (requires ffmpeg; saves to ./videos/).
  --steps        Number of policy steps to run (default 500).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 1 — Register task BEFORE any Isaac imports.
# ---------------------------------------------------------------------------
import franka_5cube_stack.camera_once  # noqa: F401

# ---------------------------------------------------------------------------
# Step 2 — Isaac Sim app.
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher  # type: ignore[import]

parser = argparse.ArgumentParser(description="Play Franka camera-once reach policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
parser.add_argument("--num_envs",   type=int, default=16)
parser.add_argument("--steps",      type=int, default=500)
parser.add_argument("--video",      action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Step 3 — Post-sim imports.
# ---------------------------------------------------------------------------
import os
import torch
import gymnasium as gym
import numpy as np

from rsl_rl.runners import OnPolicyRunner  # type: ignore[import]

from franka_5cube_stack.camera_once.agents.rsl_rl_ppo_cfg import (
    FrankaCameraOnceReachPPORunnerCfg,
)

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
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    runner_cfg = FrankaCameraOnceReachPPORunnerCfg()
    runner_cfg.device = str(args.device) if hasattr(args, "device") else "cuda:0"

    render_mode = "rgb_array" if args.video else "human"
    env = gym.make(
        "Isaac-Stack-5Cube-Franka-CameraOnce-Reach-RSLRL-v0",
        num_envs=args.num_envs,
        render_mode=render_mode,
    )

    if args.video:
        import datetime
        video_dir = os.path.abspath("videos")
        os.makedirs(video_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            name_prefix=f"reach_ppo_{stamp}",
            episode_trigger=lambda _ep: True,
        )

    env_wrapped = RslRlVecEnvWrapper(env)

    # Build runner and load checkpoint weights into actor.
    runner = OnPolicyRunner(
        env_wrapped,
        runner_cfg.to_dict() if hasattr(runner_cfg, "to_dict") else vars(runner_cfg),
        log_dir=None,
        device=runner_cfg.device,
    )
    runner.load(checkpoint_path)

    # Switch actor to eval mode and disable gradient computation.
    policy = runner.get_inference_policy(device=runner_cfg.device)

    obs, _ = env_wrapped.reset()
    total_reward = 0.0

    for step in range(args.steps):
        with torch.no_grad():
            action = policy(obs)
        obs, reward, dones, extras = env_wrapped.step(action)
        total_reward += float(reward.mean().item())

        if (step + 1) % 100 == 0:
            print(
                f"step {step + 1:4d}/{args.steps}  "
                f"mean_reward={total_reward / (step + 1):.4f}"
            )

    print(f"\nMean reward over {args.steps} steps: {total_reward / args.steps:.4f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
