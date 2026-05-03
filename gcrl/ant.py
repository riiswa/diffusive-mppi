# mujoco_playground_envs/ant.py
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

_XML_PATH = Path(__file__).parent / "assets" / "ant.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,       # n_frames=5 * sim_dt=0.01
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        impl="warp",
        naconmax=50_000,
        njmax=100,
    )


class Ant(mjx_env.MjxEnv):
    """Ant goal-reaching environment (sparse reward)."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        goal_distance: float = 10.0,
        randomize_start: bool = False,
    ):
        super().__init__(config, config_overrides)

        self._goal_distance = goal_distance
        self._randomize_start = randomize_start

        self._xml_path = str(_XML_PATH)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        # The target body uses 2 slide joints (x, y) appended at the end of qpos/qvel.
        self._target_qpos_ndof = 2
        self._target_qvel_ndof = 2
        # Torso is body index 1 (0 = world).
        self._torso_body_id = 1
        self._healthy_z_min = 0.2
        self._healthy_z_max = 1.0
        # GCRL bookkeeping
        self.state_dim = 29         # obs WITHOUT appended goal
        self.goal_indices = jp.array([0, 1])   # ant x, y in obs
        self.goal_reach_thresh = 0.5

    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        noise = 0.1
        qpos = jp.array(self._mj_model.qpos0)
        qpos = qpos + noise * jax.random.uniform(
            rng1, (self._mj_model.nq,), minval=-1.0, maxval=1.0
        )
        qvel = noise * jax.random.normal(rng2, (self._mj_model.nv,))

        target = self._random_target(rng3)
        qpos = qpos.at[-self._target_qpos_ndof:].set(target)
        qvel = qvel.at[-self._target_qvel_ndof:].set(0.0)

        if self._randomize_start:
            rng, rng4 = jax.random.split(rng)
            start = target + self._random_target(rng4)
            qpos = qpos.at[:2].set(start)

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
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        obs = self._get_obs(data)

        dist = jp.linalg.norm(obs[:2] - obs[-2:])
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
        # Mirrors jaxgcrl Ant._get_obs exactly.
        qpos = data.qpos[: -self._target_qpos_ndof]
        qvel = data.qvel[: -self._target_qvel_ndof]
        target_pos = data.qpos[-self._target_qpos_ndof:]  # slide joints == world xy
        return jp.concatenate([qpos, qvel, target_pos])

    def _random_target(self, rng: jax.Array) -> jax.Array:
        ang = jp.pi * 2.0 * jax.random.uniform(rng)
        return jp.array([
            self._goal_distance * jp.cos(ang),
            self._goal_distance * jp.sin(ang),
        ])

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