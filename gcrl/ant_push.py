# mujoco_playground_envs/ant_push.py
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

_XML_PATH = Path(__file__).parent / "assets" / "ant_push.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        impl="warp",
        naconmax=200_000,
        njmax=150,
    )


class AntPush(mjx_env.MjxEnv):
    """Ant push — push a movable block to a target (sparse reward).

    ant_push.xml structure (from jaxgcrl):
      - Ant root free joint: nq=7, nv=6
      - 8 hinge joints:      nq=8, nv=8
      - Movable block:       3 slide joints (x, y, z) nq=3, nv=3
      - Target marker:       2 slide joints (x, y)    nq=2, nv=2
    Total: nq=20, nv=19
    The last 5 qpos elements are [movable_x, movable_y, movable_z, target_x, target_y],
    which matches the original [:-5] slicing.
    Adjust if your XML differs.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        self._xml_path = str(_XML_PATH)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        # See docstring — adjust these if your XML differs.
        self._extra_qpos_ndof = 5   # movable(3) + target(2)
        self._extra_qvel_ndof = 5
        self._torso_body_id = 1
        self._healthy_z_min = 0.2
        self._healthy_z_max = 1.0

        try:
            self._movable_body_id = self._mj_model.body("movable").id
        except Exception:
            self._movable_body_id = self._mj_model.nbody - 2
        self._target_body_id = self._mj_model.nbody - 1

        self.state_dim = 31
        self.goal_indices = jp.array([0, 1])
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
        qpos = qpos.at[-2:].set(target)
        qvel = qvel.at[-self._extra_qvel_ndof:].set(0.0)

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
        qpos = data.qpos[: -self._extra_qpos_ndof]
        qvel = data.qvel[: -self._extra_qvel_ndof]
        object_pos = data.xpos[self._movable_body_id, :2]
        target_pos = data.xpos[self._target_body_id, :2]
        return jp.concatenate([qpos, qvel, object_pos, target_pos])

    def _random_target(self, rng: jax.Array) -> jax.Array:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        base = jp.array([16.0, 8.0])
        target = jax.random.permutation(rng1, base)
        noise = 2.0 * jax.random.uniform(rng2, shape=(2,))
        return target + noise

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