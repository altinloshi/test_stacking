"""Camera-once acquisition module.

Design contract
---------------
* ``run_camera_once`` is registered as an Isaac Lab **event term** (mode="reset").
  It fires for every env that is being reset and captures the ground-truth rigid-body
  poses of the five cubes.  In a real deployment the same buffer would be written by
  an actual camera pipeline; the simulator just provides ground truth here.
* After the event fires, the locked poses are **frozen** until the next reset for those
  environment indices.  The PPO policy never sees the live poses — it only reads the
  frozen snapshot, intentionally mimicking "observe once, act blind".
* Module-level dictionaries keyed by ``id(env)`` provide per-env-instance state that
  survives across event and observation calls without touching any IsaacLab core file.

Public API
----------
run_camera_once(env, env_ids)       — event term (called on reset)
get_locked_cube_poses(env)          — flat tensor (num_envs, NUM_CUBES*7)
get_active_cube_idx(env)            — int tensor  (num_envs,)
get_hover_target_pos(env)           — float tensor (num_envs, 3)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # torch is imported lazily so that CUBE_NAMES / constants can be imported
    # without a full GPU/torch installation (e.g. in CI smoke tests).
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CUBES: int = 5
CUBE_NAMES: tuple[str, ...] = ("cube_1", "cube_2", "cube_3", "cube_4", "cube_5")

#: Metres above cube centre that defines the hover goal.
HOVER_OFFSET_Z: float = 0.12

# ---------------------------------------------------------------------------
# Module-level per-env buffers  (env_object_id -> tensor)
# ---------------------------------------------------------------------------

#: Frozen cube poses, shape (num_envs, NUM_CUBES, 7)  [pos_local(3) | quat_wxyz(4)]
_LOCKED_POSES: "dict[int, torch.Tensor]" = {}

#: Active cube index per environment, shape (num_envs,)
_ACTIVE_CUBE_IDX: "dict[int, torch.Tensor]" = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _key(env: "ManagerBasedRLEnv") -> int:
    return id(env)


def _ensure_buffers(env: "ManagerBasedRLEnv") -> None:
    """Lazily allocate per-env buffers on first call."""
    import torch  # lazy — only needed at runtime, not at import time

    k = _key(env)
    if k not in _LOCKED_POSES:
        _LOCKED_POSES[k] = torch.zeros(
            env.num_envs, NUM_CUBES, 7, device=env.device, dtype=torch.float32
        )
        # Default quaternion: identity (w=1, xyz=0)
        _LOCKED_POSES[k][..., 3] = 1.0
        _ACTIVE_CUBE_IDX[k] = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )


# ---------------------------------------------------------------------------
# Public event term
# ---------------------------------------------------------------------------


def run_camera_once(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
) -> None:
    """Isaac Lab event term — fires once per reset for ``env_ids``.

    Reads the current ground-truth rigid-body state of every cube in the scene
    and writes env-local (i.e. relative to ``env.scene.env_origins``) positions
    plus world-frame quaternions into the frozen buffer for the resetting envs.

    A new active-cube index is sampled uniformly for each resetting environment.

    Args:
        env:     The manager-based RL environment.
        env_ids: 1-D long tensor of environment indices being reset this step.
    """
    import torch                        # lazy import
    import torch.nn.functional as F    # lazy import

    _ensure_buffers(env)
    k = _key(env)
    n = env_ids.numel()

    for cube_idx, name in enumerate(CUBE_NAMES):
        obj = env.scene[name]

        # pos_w / quat_w are world-frame tensors of shape (num_envs, 3/4).
        pos_w: torch.Tensor = obj.data.root_pos_w[env_ids]    # (n, 3)
        quat_w: torch.Tensor = obj.data.root_quat_w[env_ids]  # (n, 4) wxyz

        # Convert to env-local frame so observations are translation-invariant.
        pos_local = pos_w - env.scene.env_origins[env_ids]    # (n, 3)

        # Normalise quaternion defensively (simulation should keep it unit, but
        # floating-point drift can accumulate over long runs).
        quat_norm = F.normalize(quat_w, p=2, dim=-1)

        _LOCKED_POSES[k][env_ids, cube_idx, :3] = pos_local
        _LOCKED_POSES[k][env_ids, cube_idx, 3:] = quat_norm

    # Randomly pick a new active cube for every resetting environment.
    _ACTIVE_CUBE_IDX[k][env_ids] = torch.randint(
        0, NUM_CUBES, (n,), device=env.device, dtype=torch.long
    )


# ---------------------------------------------------------------------------
# Public accessor functions (called by mdp.py observation terms)
# ---------------------------------------------------------------------------


def get_locked_cube_poses(env: "ManagerBasedRLEnv") -> "torch.Tensor":
    """Return frozen cube poses as a flat vector.

    Returns:
        Tensor of shape ``(num_envs, NUM_CUBES * 7)`` containing
        [pos_local | quat_wxyz] for each cube, concatenated along the last
        dimension.  Values are constant between resets.
    """
    _ensure_buffers(env)
    return _LOCKED_POSES[_key(env)].reshape(env.num_envs, NUM_CUBES * 7)


def get_active_cube_idx(env: "ManagerBasedRLEnv") -> "torch.Tensor":
    """Return the active (target) cube index per env.

    Returns:
        Long tensor of shape ``(num_envs,)`` in ``[0, NUM_CUBES)``.
    """
    _ensure_buffers(env)
    return _ACTIVE_CUBE_IDX[_key(env)]


def get_active_cube_idx_onehot(env: "ManagerBasedRLEnv") -> "torch.Tensor":
    """Return one-hot encoding of the active cube index.

    Returns:
        Float tensor of shape ``(num_envs, NUM_CUBES)``.
    """
    import torch  # lazy import

    idx = get_active_cube_idx(env)
    onehot = torch.zeros(env.num_envs, NUM_CUBES, device=env.device, dtype=torch.float32)
    onehot.scatter_(1, idx.unsqueeze(-1), 1.0)
    return onehot


def get_hover_target_pos(env: "ManagerBasedRLEnv") -> "torch.Tensor":
    """Return the hover waypoint above the active cube (env-local frame).

    The hover target is defined as the locked position of the active cube
    plus ``HOVER_OFFSET_Z`` metres in the +Z direction.

    Returns:
        Float tensor of shape ``(num_envs, 3)``.
    """
    import torch  # lazy import

    _ensure_buffers(env)
    k = _key(env)
    poses = _LOCKED_POSES[k]        # (num_envs, NUM_CUBES, 7)
    active_idx = _ACTIVE_CUBE_IDX[k]  # (num_envs,)

    row_idx = torch.arange(env.num_envs, device=env.device)
    active_pos = poses[row_idx, active_idx, :3].clone()  # (num_envs, 3)
    active_pos[:, 2] += HOVER_OFFSET_Z
    return active_pos
