# mujoco_playground_envs/humanoid.py
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

_XML_PATH = Path(__file__).parent / "assets" / "humanoid.xml"

# Fixed z-height of the goal marker (same as jaxgcrl).
TARGET_Z_COORD = 1.25


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.025,       # n_frames=5 * sim_dt=0.005
        sim_dt=0.005,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        impl="warp",
        naconmax=500_000,
        njmax=250,
    )


class Humanoid(mjx_env.MjxEnv):
    """Humanoid goal-reaching environment (sparse reward).

    Observation note
    ----------------
    The original jaxgcrl humanoid builds a 268-dim obs using Brax-internal
    composite inertia/velocity tensors.  Here we use the MuJoCo equivalents:
      data.cinert  (nbody x 10) — composite inertia in COM frame
      data.cvel    (nbody x 6)  — body spatial velocities

    The resulting obs dim will differ from 268; update `state_dim` after
    a first print of obs.shape.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        min_goal_dist: float = 1.0,
        max_goal_dist: float = 5.0,
    ):
        super().__init__(config, config_overrides)

        self._min_goal_dist = min_goal_dist
        self._max_goal_dist = max_goal_dist

        self._xml_path = str(_XML_PATH)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        # Target body: last 2 qpos DOFs (x, y slide joints)
        self._target_qpos_ndof = 2
        self._target_qvel_ndof = 2
        self._torso_body_id = 1   # adjust if needed
        self._healthy_z_min = 1.0
        self._healthy_z_max = 2.0
        # GCRL: goal is 3-D position (obs[:3] = humanoid xyz)
        self.goal_indices = jp.array([0, 1, 2])
        self.goal_reach_thresh = 0.5

    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        qpos = jp.array(self._mj_model.qpos0)
        qvel = jp.zeros(self._mj_model.nv)

        target = self._random_target(rng3)
        qpos = qpos.at[-self._target_qpos_ndof:].set(target)
        qvel = qvel.at[-self._target_qvel_ndof:].set(0.0)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = data.replace(qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        metrics = {
            "dist": jp.zeros(()),
            "success": jp.zeros(()),
            "success_easy": jp.zeros(()),
        }
        obs = self._get_obs(data)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, {"rng": rng})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Scale action from [-1,1] to actuator limits (mirrors jaxgcrl)
        action_min = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
        action_max = jp.array(self._mj_model.actuator_ctrlrange[:, 1])
        scaled_action = (action + 1.0) * (action_max - action_min) * 0.5 + action_min

        data = mjx_env.step(self.mjx_model, state.data, scaled_action, self.n_substeps)
        obs = self._get_obs(data)

        dist = jp.linalg.norm(obs[:3] - obs[-3:])
        success = jp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jp.array(dist < 2.0, dtype=float)
        reward = success

        is_unhealthy = (
            (data.xpos[self._torso_body_id, 2] < self._healthy_z_min)
            | (data.xpos[self._torso_body_id, 2] > self._healthy_z_max)
        )
        done = is_unhealthy.astype(float)

        state.metrics.update(dist=dist, success=success, success_easy=success_easy)
        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    # ------------------------------------------------------------------
    def _get_obs(self, data: mjx.Data) -> jax.Array:
        # Full state (mirrors jaxgcrl structure with MJX equivalents)
        position = data.qpos                    # nq  (includes target at end)
        velocity = data.qvel                    # nv
        com_inertia = data.cinert.ravel()       # nbody * 10
        com_velocity = data.cvel.ravel()        # nbody * 6
        qfrc_actuator = data.qfrc_actuator      # nu

        target_pos = data.qpos[-self._target_qpos_ndof:]       # (2,)
        target_z   = jp.array([TARGET_Z_COORD])                # (1,)

        return jp.concatenate([
            position,
            velocity,
            com_inertia,
            com_velocity,
            qfrc_actuator,
            target_pos,
            target_z,
        ])

    def _random_target(self, rng: jax.Array) -> jax.Array:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = jax.random.uniform(rng1, minval=self._min_goal_dist, maxval=self._max_goal_dist)
        ang = jp.pi * 2.0 * jax.random.uniform(rng2)
        return jp.array([dist * jp.cos(ang), dist * jp.sin(ang)])

    # ------------------------------------------------------------------
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model