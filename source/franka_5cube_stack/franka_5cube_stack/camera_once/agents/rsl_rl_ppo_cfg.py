"""RSL-RL on-policy PPO runner configuration for the camera-once reach task.

The class :class:`FrankaCameraOnceReachPPORunnerCfg` is referenced by the
Gymnasium registration in :mod:`franka_5cube_stack.camera_once` via the key
``rsl_rl_cfg_entry_point``.

Import hierarchy
----------------
``isaaclab_rl.rsl_rl`` is tried first (Isaac Lab ≥ 1.x with the split
``isaaclab_rl`` package).  Older single-package installs may expose the same
classes under ``isaaclab_tasks.utils.wrappers.rsl_rl``.  A plain-dataclass
fallback is defined so that the file is importable even when neither is
available (e.g. in CI environments without a full Isaac Lab install).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Import RSL-RL base config classes with graceful fallbacks
# ---------------------------------------------------------------------------

try:
    from isaaclab_rl.rsl_rl import (  # type: ignore[import]
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
    )
    _HAVE_RSL_RL_CFG = True
except ImportError:
    try:
        from isaaclab_tasks.utils.wrappers.rsl_rl import (  # type: ignore[import]
            RslRlOnPolicyRunnerCfg,
            RslRlPpoActorCriticCfg,
            RslRlPpoAlgorithmCfg,
        )
        _HAVE_RSL_RL_CFG = True
    except ImportError:
        # ---------------------------------------------------------------------------
        # Minimal stub so the file is parseable in lint / smoke-test contexts that
        # lack a full Isaac Lab installation.  The runner will raise at runtime if
        # the stubs are used to actually train, which gives a clear error message.
        # ---------------------------------------------------------------------------
        _HAVE_RSL_RL_CFG = False

        @configclass
        class RslRlPpoActorCriticCfg:  # type: ignore[no-redef]
            init_noise_std: float = 1.0
            actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
            critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
            activation: str = "elu"

        @configclass
        class RslRlPpoAlgorithmCfg:  # type: ignore[no-redef]
            value_loss_coef: float = 1.0
            use_clipping: bool = True
            clip_param: float = 0.2
            entropy_coef: float = 0.005
            num_learning_epochs: int = 5
            num_mini_batches: int = 4
            learning_rate: float = 1e-3
            schedule: str = "adaptive"
            gamma: float = 0.99
            lam: float = 0.95
            desired_kl: float = 0.01
            max_grad_norm: float = 1.0

        @configclass
        class RslRlOnPolicyRunnerCfg:  # type: ignore[no-redef]
            num_steps_per_env: int = 24
            max_iterations: int = 1500
            save_interval: int = 50
            experiment_name: str = ""
            run_name: str = ""
            resume: bool = False
            load_run: str = ".*"
            load_checkpoint: str = "model_.*.pt"
            logger: str = "tensorboard"
            neptune_project: str = "isaaclab"
            wandb_project: str = "isaaclab"
            device: str = "cuda:0"
            seed: int = 42
            policy: RslRlPpoActorCriticCfg = field(
                default_factory=RslRlPpoActorCriticCfg
            )
            algorithm: RslRlPpoAlgorithmCfg = field(
                default_factory=RslRlPpoAlgorithmCfg
            )


# ---------------------------------------------------------------------------
# Task-specific network configs
# ---------------------------------------------------------------------------


@configclass
class _PolicyNetCfg(RslRlPpoActorCriticCfg):
    """Actor-critic network for the 71-dim observation / 9-dim action space."""

    init_noise_std: float = 1.0
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"


@configclass
class _AlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO hyper-parameters tuned for a 71-dim obs space and 25 Hz control."""

    value_loss_coef: float = 1.0
    use_clipping: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# Public runner configuration
# ---------------------------------------------------------------------------


@configclass
class FrankaCameraOnceReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL ``OnPolicyRunner`` configuration for the reach skill.

    Typical training invocation::

        from franka_5cube_stack.camera_once.agents.rsl_rl_ppo_cfg import (
            FrankaCameraOnceReachPPORunnerCfg,
        )
        runner_cfg = FrankaCameraOnceReachPPORunnerCfg()
    """

    # ----- Runner -----
    seed:            int   = 42
    device:          str   = "cuda:0"
    num_steps_per_env: int = 24       # roll-out length per env per update
    max_iterations:  int   = 1500
    save_interval:   int   = 50

    # ----- Logging -----
    experiment_name: str = "franka_camera_once_reach"
    run_name:        str = ""
    logger:          str = "tensorboard"  # "wandb" | "neptune" | "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project:   str = "isaaclab"

    # ----- Checkpoint resume -----
    resume:          bool = False
    load_run:        str  = ".*"
    load_checkpoint: str  = "model_.*.pt"

    # ----- Network and algorithm -----
    policy:    _PolicyNetCfg = field(default_factory=_PolicyNetCfg)
    algorithm: _AlgorithmCfg = field(default_factory=_AlgorithmCfg)
