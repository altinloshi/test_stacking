"""MDP building blocks for the camera-once reach task.

Every public function in this module matches the Isaac Lab term-function
signature for the corresponding manager type:

  Observation term : ``func(env) -> Tensor(num_envs, dim)``
  Reward term      : ``func(env) -> Tensor(num_envs,)``
  Termination term : ``func(env) -> BoolTensor(num_envs,)``
  Event term       : ``func(env, env_ids) -> None``

Observation vector composition (total 71 dims)
----------------------------------------------
  joint_pos          9   panda_joint[1-7] + panda_finger_joint[1-2]
  joint_vel          9   same joints
  ee_pos             3   end-effector position, env-local frame
  ee_quat            4   end-effector orientation, wxyz
  locked_cube_poses 35   5 cubes × 7  (pos_local + quat_wxyz), frozen per reset
  active_cube_onehot 5   one-hot over 5 cubes
  hover_target_pos   3   active-cube pos + HOVER_OFFSET_Z, env-local
  ee_to_hover        3   hover_target_pos − ee_pos
                   ---
                    71

Action space (9 dims)
---------------------
  Joint position targets: panda_joint[1-7] + panda_finger_joint[1-2]
  Controlled via ``JointPositionActionCfg``, scale 0.5, relative offset from
  default pose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # required at runtime — only imported when actual env is running

from . import camera_acquisition as ca

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Name of the robot articulation in the scene.
_ROBOT_NAME: str = "robot"

# End-effector body name on the Franka Panda.
_EE_BODY_NAME: str = "panda_hand"

# Cached body index per env-instance (avoids repeated string lookup).
_ee_body_idx_cache: dict[int, int] = {}


def _get_ee_body_idx(env: "ManagerBasedRLEnv") -> int:
    k = id(env)
    if k not in _ee_body_idx_cache:
        robot = env.scene[_ROBOT_NAME]
        body_names: list[str] = robot.data.body_names
        idx = body_names.index(_EE_BODY_NAME)
        _ee_body_idx_cache[k] = idx
    return _ee_body_idx_cache[k]


# ---------------------------------------------------------------------------
# Observation terms
# ---------------------------------------------------------------------------


def joint_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Joint positions of all 9 Franka joints (arm + gripper), shape (N, 9)."""
    robot = env.scene[_ROBOT_NAME]
    return robot.data.joint_pos  # (num_envs, num_joints)


def joint_vel(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Joint velocities of all 9 Franka joints, shape (N, 9)."""
    robot = env.scene[_ROBOT_NAME]
    return robot.data.joint_vel  # (num_envs, num_joints)


def ee_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """End-effector position in env-local frame, shape (N, 3).

    Computed from the ``panda_hand`` body world position minus the
    per-env world origin so the value is invariant to env tiling.
    """
    robot = env.scene[_ROBOT_NAME]
    ee_idx = _get_ee_body_idx(env)
    pos_w = robot.data.body_pos_w[:, ee_idx, :]  # (N, 3)
    return pos_w - env.scene.env_origins          # (N, 3)


def ee_quat(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """End-effector orientation as wxyz quaternion, shape (N, 4)."""
    robot = env.scene[_ROBOT_NAME]
    ee_idx = _get_ee_body_idx(env)
    return robot.data.body_quat_w[:, ee_idx, :]   # (N, 4)


def locked_cube_poses_flat(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Frozen cube poses (captured once per reset), shape (N, 35).

    Reads from the :mod:`camera_acquisition` frozen buffer.  Values do NOT
    change between resets regardless of cube motion during the episode.
    """
    return ca.get_locked_cube_poses(env)


def active_cube_onehot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-hot indicator of which cube is the active reach target, shape (N, 5)."""
    return ca.get_active_cube_idx_onehot(env)


def hover_target_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Hover goal position above active cube (env-local), shape (N, 3)."""
    return ca.get_hover_target_pos(env)


def ee_to_hover_target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Displacement vector from end-effector to hover goal, shape (N, 3).

    Positive values mean the EE must move in that direction to reach the goal.
    """
    return ca.get_hover_target_pos(env) - ee_pos(env)


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def reach_hover_target_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Dense negative-distance reward for reaching the hover goal.

    Returns ``−‖ee_pos − hover_target‖₂``, clipped to ``[−1, 0]`` so the
    policy always receives a bounded, non-positive signal.

    Shape: ``(num_envs,)``
    """
    dist = torch.norm(ee_to_hover_target(env), dim=-1)  # (N,)
    return -dist.clamp(max=1.0)


def hover_success_bonus(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.02,
    bonus: float = 1.0,
) -> torch.Tensor:
    """Sparse bonus when EE is within ``threshold`` metres of hover goal.

    Shape: ``(num_envs,)``
    """
    dist = torch.norm(ee_to_hover_target(env), dim=-1)
    return (dist < threshold).float() * bonus


def action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalise large changes in consecutive actions to encourage smoothness.

    Requires the action manager to expose ``prev_action`` (standard in
    Isaac Lab's ``ActionManager``).

    Shape: ``(num_envs,)``
    """
    action = env.action_manager.action          # (N, act_dim)
    prev   = env.action_manager.prev_action     # (N, act_dim)
    return -torch.norm(action - prev, dim=-1)


# ---------------------------------------------------------------------------
# Termination terms
# ---------------------------------------------------------------------------


def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Terminate episode when the episode length exceeds the configured limit.

    Isaac Lab's ``TimeOutTermCfg`` handles this natively; this function is
    provided as a drop-in replacement / explicit override.

    Shape: ``(num_envs,)`` bool
    """
    return env.episode_length_buf >= env.max_episode_length


def hover_reached(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.02,
) -> torch.Tensor:
    """Terminate (successfully) once the EE is within ``threshold`` of hover goal.

    Shape: ``(num_envs,)`` bool
    """
    dist = torch.norm(ee_to_hover_target(env), dim=-1)
    return dist < threshold


# ---------------------------------------------------------------------------
# Event terms
# ---------------------------------------------------------------------------


def reset_robot_to_default(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
) -> None:
    """Reset Franka joints to the default (home) configuration.

    Uses the articulation's own ``reset()`` mechanism so that joint
    positions, velocities and accelerations are all zeroed / set to defaults.
    """
    robot = env.scene[_ROBOT_NAME]
    robot.reset(env_ids)


def randomize_cube_positions(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (0.35, 0.65),
    y_range: tuple[float, float] = (-0.30, 0.30),
    z_val:   float = 0.025,
) -> None:
    """Scatter the five cubes to random (x, y) positions on the workspace surface.

    Positions are sampled uniformly within ``x_range`` × ``y_range`` in the
    env-local frame, then converted to world frame by adding ``env_origins``.
    A minimum inter-cube separation of 0.08 m is enforced via rejection sampling
    (up to 20 attempts; falls back to last sample on failure).

    Args:
        env:      The RL environment.
        env_ids:  Indices of the environments being reset.
        x_range:  (min_x, max_x) in env-local metres.
        y_range:  (min_y, max_y) in env-local metres.
        z_val:    Fixed spawn height (half cube edge = 0.025 m).
    """
    n = env_ids.numel()
    dev = env.device
    origins = env.scene.env_origins[env_ids]  # (n, 3)

    min_sep: float = 0.08
    max_tries: int = 20

    for cube_idx, name in enumerate(ca.CUBE_NAMES):
        obj = env.scene[name]

        # Rejection sampling to avoid cube overlap.
        for _ in range(max_tries):
            lx = torch.rand(n, 1, device=dev) * (x_range[1] - x_range[0]) + x_range[0]
            ly = torch.rand(n, 1, device=dev) * (y_range[1] - y_range[0]) + y_range[0]
            lz = torch.full((n, 1), z_val, device=dev)
            pos_local = torch.cat([lx, ly, lz], dim=-1)  # (n, 3)

            if cube_idx == 0:
                break  # first cube has no prior cubes to avoid

            # Check against already-placed cubes (positions stored in scene).
            collision = torch.zeros(n, dtype=torch.bool, device=dev)
            for prev_name in ca.CUBE_NAMES[:cube_idx]:
                prev_obj = env.scene[prev_name]
                prev_local = prev_obj.data.root_pos_w[env_ids] - origins
                dist_xy = torch.norm(pos_local[:, :2] - prev_local[:, :2], dim=-1)
                collision |= dist_xy < min_sep
            if not collision.any():
                break

        pos_world = pos_local + origins  # (n, 3)

        # Preserve current quaternion (cubes spawn axis-aligned).
        quat_w = obj.data.root_quat_w[env_ids]

        obj.write_root_pose_to_sim(
            torch.cat([pos_world, quat_w], dim=-1),  # (n, 7)
            env_ids,
        )
