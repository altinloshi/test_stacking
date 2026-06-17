"""camera_once sub-package — Gymnasium task registration.

Importing this module (or any sub-module) causes the Isaac Lab task
``Isaac-Stack-5Cube-Franka-CameraOnce-Reach-RSLRL-v0`` to be registered
with Gymnasium **before** any Hydra / RSL-RL runner resolves the task name.

Architecture summary
--------------------
* Camera fires **once per env reset** (event term ``run_camera_once``).
* Cube world positions are captured into a frozen per-env buffer and do NOT
  change until the next reset.
* The PPO policy receives a low-dimensional observation vector only; no RGB
  frames are consumed at every step.
* The first learned skill is "reach to hover position above active cube".
"""

import gymnasium as gym

gym.register(
    id="Isaac-Stack-5Cube-Franka-CameraOnce-Reach-RSLRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "franka_5cube_stack.camera_once.env_cfg:FrankaCameraOnceReachEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "franka_5cube_stack.camera_once.agents.rsl_rl_ppo_cfg"
            ":FrankaCameraOnceReachPPORunnerCfg"
        ),
    },
)
