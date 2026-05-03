# mujoco_playground_envs/ant_ball.py
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

_XML_PATH = Path(__file__).parent / "assets" / "ant_ball.xml"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        impl="warp",
        naconmax=100_000,
        njmax=150,
    )


class AntBall(mjx_env.MjxEnv):
    """Ant soccer — push a ball to a target (sparse reward)."""

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
        # ant_ball.xml: target body (2 slides) + ball/object body (2 slides)
        # → last 4 qpos / qvel DOFs are [object_x, object_y, target_x, target_y]
        self._obj_target_qpos_ndof = 4   # object(2) + target(2)
        self._obj_target_qvel_ndof = 4
        self._torso_body_id = 1
        self._healthy_z_min = 0.2
        self._healthy_z_max = 1.0
        # The object body comes before the target body in the link list.
        # Adjust these two indices to match your XML if needed.
        self._object_body_name = "object"
        try:
            self._object_body_id = self._mj_model.body(self._object_body_name).id
        except Exception:
            # Fallback: second-to-last body
            self._object_body_id = self._mj_model.nbody - 2
        self._target_body_id = self._mj_model.nbody - 1

        self.state_dim = 31
        self.goal_indices = jp.array([29, 30])   # object xy inside obs
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

        target, obj = self._random_target_and_ball(rng3)
        # Layout at end of qpos: [obj_x, obj_y, target_x, target_y]
        qpos = qpos.at[-4:-2].set(obj)
        qpos = qpos.at[-2:].set(target)
        qvel = qvel.at[-4:].set(0.0)

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

        # Distance between ball (goal_indices) and target (last 2)
        dist = jp.linalg.norm(obs[-2:] - obs[-4:-2])
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
        # Mirrors jaxgcrl AntBall._get_obs
        qpos = data.qpos[:-4]
        qvel = data.qvel[:-4]
        object_pos = data.xpos[self._object_body_id, :2]
        target_pos = data.xpos[self._target_body_id, :2]
        return jp.concatenate([qpos, qvel, object_pos, target_pos])

    def _random_target_and_ball(self, rng: jax.Array):
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = 5.0
        ang = jp.pi * 2.0 * jax.random.uniform(rng1)
        target = jp.array([dist * jp.cos(ang), dist * jp.sin(ang)])
        ang_obj = jp.pi * 2.0 * jax.random.uniform(rng2)
        obj = target * 0.2 + jp.array([jp.cos(ang_obj), jp.sin(ang_obj)])
        return target, obj

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