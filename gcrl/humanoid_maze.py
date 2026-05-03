# mujoco_playground_envs/humanoid_maze.py
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

# Reuse maze layouts from ant_maze (or duplicate here for independence)
from ant_maze import (
    MAZE_LAYOUTS, MAZE_HEIGHT, _find_starts, _find_goals
)

TARGET_Z_COORD = 1.25
_BASE_XML = Path(__file__).parent / "assets" / "humanoid_maze.xml"


def _make_humanoid_maze_xml(maze_layout_name: str, maze_size_scaling: float):
    layout = MAZE_LAYOUTS[maze_layout_name]
    possible_starts = _find_starts(layout, maze_size_scaling)
    possible_goals  = _find_goals(layout, maze_size_scaling)

    tree = ET.parse(str(_BASE_XML))
    worldbody = tree.find(".//worldbody")

    for i, row in enumerate(layout):
        for j, cell in enumerate(row):
            if cell == 1:
                ET.SubElement(
                    worldbody, "geom",
                    name=f"block_{i}_{j}",
                    pos=(f"{i*maze_size_scaling:.4f} "
                         f"{j*maze_size_scaling:.4f} "
                         f"{MAZE_HEIGHT/2*maze_size_scaling:.4f}"),
                    size=(f"{0.5*maze_size_scaling:.4f} "
                          f"{0.5*maze_size_scaling:.4f} "
                          f"{MAZE_HEIGHT/2*maze_size_scaling:.4f}"),
                    type="box", material="",
                    contype="1", conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    xml_string = ET.tostring(tree.getroot(), encoding="unicode")
    return xml_string, possible_starts, possible_goals


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.025,
        sim_dt=0.005,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        impl="warp",
        naconmax=200_000,
        njmax=300,
    )


class HumanoidMaze(mjx_env.MjxEnv):
    """HumanoidMaze — sparse reward, goal-conditioned."""

    def __init__(
        self,
        maze_layout_name: str = "u_maze",
        maze_size_scaling: float = 2.0,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        xml_string, possible_starts, possible_goals = _make_humanoid_maze_xml(
            maze_layout_name, maze_size_scaling
        )
        self.possible_starts = possible_starts
        self.possible_goals  = possible_goals

        self._xml_path = f"humanoid_maze_{maze_layout_name}"
        self._mj_model = mujoco.MjModel.from_xml_string(xml_string)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        self._target_qpos_ndof = 2
        self._target_qvel_ndof = 2
        self._torso_body_id = 1
        self._healthy_z_min = 1.0
        self._healthy_z_max = 2.0
        self.goal_indices = jp.array([0, 1, 2])
        self.goal_reach_thresh = 0.5

    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        qpos = jp.array(self._mj_model.qpos0)
        qvel = jp.zeros(self._mj_model.nv)

        start  = self._random_start(rng2)
        target = self._random_target(rng3)
        qpos = qpos.at[:2].set(start)
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
        position     = data.qpos
        velocity     = data.qvel
        com_inertia  = data.cinert.ravel()
        com_velocity = data.cvel.ravel()
        qfrc_actuator = data.qfrc_actuator
        target_pos   = data.qpos[-self._target_qpos_ndof:]
        target_z     = jp.array([TARGET_Z_COORD])
        return jp.concatenate([
            position, velocity,
            com_inertia, com_velocity,
            qfrc_actuator,
            target_pos, target_z,
        ])

    def _random_target(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))[0]
        return self.possible_goals[idx]

    def _random_start(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))[0]
        return self.possible_starts[idx]

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


class HumanoidUMaze(HumanoidMaze):
    def __init__(self, **kwargs):
        super().__init__(maze_layout_name="u_maze", **kwargs)

class HumanoidBigMaze(HumanoidMaze):
    def __init__(self, **kwargs):
        super().__init__(maze_layout_name="big_maze", **kwargs)

class HumanoidHardestMaze(HumanoidMaze):
    def __init__(self, **kwargs):
        super().__init__(maze_layout_name="hardest_maze", **kwargs)