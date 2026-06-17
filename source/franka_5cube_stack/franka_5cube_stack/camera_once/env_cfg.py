"""Environment configuration for the Franka camera-once reach task.

This file defines :class:`FrankaCameraOnceReachEnvCfg`, a
:class:`~isaaclab.envs.ManagerBasedRLEnvCfg` subclass that wires together:

* **Scene**  — Franka Panda, five coloured cubes, ground plane, lighting and
  an overhead RGB-D camera (defined for completeness; its data stream is only
  consumed by the ``run_camera_once`` event, not at every policy step).
* **Actions** — 9-DOF joint-position targets (7 arm + 2 gripper fingers).
* **Observations** — 71-dimensional flat policy vector (no RGB frames).
* **Rewards** — dense reach reward + sparse hover bonus + action-rate penalty.
* **Terminations** — episode time-out; optional success termination.
* **Events** — reset robot, randomise cube positions, run camera once.

No IsaacLab core files are modified.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import camera_acquisition as ca
from . import mdp

# ---------------------------------------------------------------------------
# Try to import the Franka asset from the standard isaaclab_assets package.
# Fall back to a minimal ArticulationCfg stub so the file is still importable
# in environments that only have isaaclab_assets partially installed.
# ---------------------------------------------------------------------------
try:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # type: ignore[import]
except ImportError:
    try:
        from isaaclab_assets import FRANKA_PANDA_CFG  # type: ignore[import]
    except ImportError:
        # Minimal stub — allows smoke-tests / linting without full assets.
        FRANKA_PANDA_CFG = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=(
                    "{ISAACLAB_ASSETS_PATH}/robots/franka/franka_panda_instanceable.usd"
                ),
                activate_contact_sensors=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.5695,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.8100,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.0368,
                    "panda_joint7": 0.7408,
                    "panda_finger_joint.*": 0.04,
                },
            ),
            actuators={},
        )


# ---------------------------------------------------------------------------
# Cube spawn helpers
# ---------------------------------------------------------------------------

#: Half-edge length of each cube in metres.
_CUBE_HALF_SIZE: float = 0.025

#: Default cube spawn positions (env-local).  The ``randomize_cube_positions``
#: event overwrites these on every reset; they serve only as the USD prim origin.
_CUBE_DEFAULTS: list[tuple[float, float, float]] = [
    (0.50,  0.00, _CUBE_HALF_SIZE),
    (0.50,  0.15, _CUBE_HALF_SIZE),
    (0.50, -0.15, _CUBE_HALF_SIZE),
    (0.40,  0.10, _CUBE_HALF_SIZE),
    (0.40, -0.10, _CUBE_HALF_SIZE),
]

#: Distinct diffuse colours for each cube (RGB in [0, 1]).
_CUBE_COLORS: list[tuple[float, float, float]] = [
    (0.85, 0.15, 0.15),  # red
    (0.15, 0.75, 0.15),  # green
    (0.15, 0.35, 0.85),  # blue
    (0.90, 0.80, 0.10),  # yellow
    (0.90, 0.45, 0.10),  # orange
]


def _make_cube_cfg(
    name: str,
    default_pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """Return a ``RigidObjectCfg`` for one 5 cm cube."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name.title().replace("_", ""),
        spawn=sim_utils.CuboidCfg(
            size=(2 * _CUBE_HALF_SIZE,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=default_pos),
    )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class FrankaCameraOnceSceneCfg(InteractiveSceneCfg):
    """Isaac Lab scene: Franka robot + 5 cubes + camera + environment assets."""

    # ----- Environment infrastructure -----

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(1.0, 1.0, 1.0),
        ),
    )

    # ----- Robot -----

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # ----- Five cubes -----

    cube_1: RigidObjectCfg = _make_cube_cfg("cube_1", _CUBE_DEFAULTS[0], _CUBE_COLORS[0])
    cube_2: RigidObjectCfg = _make_cube_cfg("cube_2", _CUBE_DEFAULTS[1], _CUBE_COLORS[1])
    cube_3: RigidObjectCfg = _make_cube_cfg("cube_3", _CUBE_DEFAULTS[2], _CUBE_COLORS[2])
    cube_4: RigidObjectCfg = _make_cube_cfg("cube_4", _CUBE_DEFAULTS[3], _CUBE_COLORS[3])
    cube_5: RigidObjectCfg = _make_cube_cfg("cube_5", _CUBE_DEFAULTS[4], _CUBE_COLORS[4])

    # ----- Overhead RGB-D camera -----
    #
    # The camera is defined here for completeness and future real-image support.
    # During training, ``run_camera_once`` reads ground-truth rigid-body state
    # rather than rendered pixels; the camera sensor is therefore not required
    # for the PPO loop but is available for debugging visualisations.

    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,  # updated on demand (once per reset via event)
        height=480,
        width=640,
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 1.20),
            rot=(0.0, math.sin(math.radians(60.0) / 2), 0.0, math.cos(math.radians(60.0) / 2)),
            convention="ros",
        ),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class FrankaCameraOnceActionsCfg:
    """9-DOF joint-position action space (7 arm joints + 2 finger joints).

    The action is a relative delta from the current joint position,
    scaled by ``scale=0.5`` radians.  Finger joints are included so the
    gripper is part of the learned policy; a higher-level planner can
    override them when required.
    """

    body_joint_pos = SceneEntityCfg(
        name="robot",
    )

    # Loaded via JointPositionActionCfg inside ManagerBasedRLEnv.
    # Defined explicitly in env_cfg to expose the action term class.
    joint_pos: object = None  # populated below after imports

    def __post_init__(self) -> None:
        # Import here to avoid circular imports at module load time.
        from isaaclab.envs.mdp.actions import JointPositionActionCfg  # type: ignore[import]

        self.joint_pos = JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_finger_joint1",
                "panda_finger_joint2",
            ],
            scale=0.5,
            use_default_offset=True,
        )


# ---------------------------------------------------------------------------
# Observations  (71 dims total)
# ---------------------------------------------------------------------------


@configclass
class FrankaCameraOncePolicyCfg(ObservationGroupCfg):
    """Flat low-dimensional observation for the PPO policy.

    Concatenation order (left to right matches tensor columns):

    ======  =====  ===============================================
    Slice   Dims   Content
    ======  =====  ===============================================
    [0:9]     9    Joint positions (arm + gripper)
    [9:18]    9    Joint velocities (arm + gripper)
    [18:21]   3    End-effector position, env-local
    [21:25]   4    End-effector quaternion, wxyz
    [25:60]  35    Frozen cube poses (5 × [pos+quat])
    [60:65]   5    Active cube one-hot
    [65:68]   3    Hover target position, env-local
    [68:71]   3    EE → hover target displacement
    ======  =====  ===============================================
    """

    enable_corruption: bool = False

    joint_pos     = ObservationTermCfg(func=mdp.joint_pos)
    joint_vel     = ObservationTermCfg(func=mdp.joint_vel)
    ee_pos        = ObservationTermCfg(func=mdp.ee_pos)
    ee_quat       = ObservationTermCfg(func=mdp.ee_quat)
    locked_cubes  = ObservationTermCfg(func=mdp.locked_cube_poses_flat)
    active_onehot = ObservationTermCfg(func=mdp.active_cube_onehot)
    hover_target  = ObservationTermCfg(func=mdp.hover_target_pos)
    ee_to_hover   = ObservationTermCfg(func=mdp.ee_to_hover_target)


@configclass
class FrankaCameraOnceObservationsCfg:
    """Observation manager configuration."""

    policy: FrankaCameraOncePolicyCfg = FrankaCameraOncePolicyCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class FrankaCameraOnceRewardsCfg:
    """Reward signal for the reach-to-hover-target skill."""

    # Dense: negative L2 distance (always ≤ 0, clipped to -1).
    reach = RewardTermCfg(
        func=mdp.reach_hover_target_reward,
        weight=2.0,
    )

    # Sparse bonus at success threshold.
    hover_bonus = RewardTermCfg(
        func=mdp.hover_success_bonus,
        weight=5.0,
        params={"threshold": 0.02, "bonus": 1.0},
    )

    # Smooth action penalty.
    action_rate = RewardTermCfg(
        func=mdp.action_rate_penalty,
        weight=-0.01,
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class FrankaCameraOnceTerminationsCfg:
    """Episode termination conditions."""

    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )

    # Uncomment to terminate on success (useful for curriculum):
    # hover_success = TerminationTermCfg(
    #     func=mdp.hover_reached,
    #     params={"threshold": 0.02},
    # )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@configclass
class FrankaCameraOnceEventsCfg:
    """Reset and initialisation events."""

    reset_robot = EventTermCfg(
        func=mdp.reset_robot_to_default,
        mode="reset",
    )

    randomize_cubes = EventTermCfg(
        func=mdp.randomize_cube_positions,
        mode="reset",
        params={
            "x_range": (0.35, 0.65),
            "y_range": (-0.30, 0.30),
            "z_val": _CUBE_HALF_SIZE,
        },
    )

    # Camera fires AFTER cubes are randomised so poses are final.
    camera_once = EventTermCfg(
        func=ca.run_camera_once,
        mode="reset",
    )


# ---------------------------------------------------------------------------
# Top-level environment config
# ---------------------------------------------------------------------------


@configclass
class FrankaCameraOnceReachEnvCfg(ManagerBasedRLEnvCfg):
    """Full configuration for the Franka 5-cube camera-once reach environment.

    Observation space : 71 dims  (see :class:`FrankaCameraOncePolicyCfg`)
    Action space      :  9 dims  (joint position targets)
    Episode length    : 200 steps × 0.02 s = 4 s per episode
    """

    # ----- Scene -----
    scene: FrankaCameraOnceSceneCfg = FrankaCameraOnceSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
    )

    # ----- MDP components -----
    actions      = FrankaCameraOnceActionsCfg()
    observations = FrankaCameraOnceObservationsCfg()
    rewards      = FrankaCameraOnceRewardsCfg()
    terminations = FrankaCameraOnceTerminationsCfg()
    events       = FrankaCameraOnceEventsCfg()

    # ----- Timing -----
    decimation:         int   = 2      # policy @ 25 Hz with 50 Hz physics
    episode_length_s:   float = 4.0    # 200 policy steps

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.dt = 0.02             # physics step: 50 Hz
        self.sim.render_interval = self.decimation

        # Observation space dimension: 9+9+3+4+35+5+3+3 = 71
        # Action space dimension: 9
        # (Isaac Lab's ManagerBasedRLEnv derives these automatically from the
        # manager configs; no explicit override needed.)
