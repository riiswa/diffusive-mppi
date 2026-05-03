# mujoco_playground_envs/ant_maze.py
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

# ── Maze layouts (identical to jaxgcrl) ──────────────────────────────────────
RESET = R = "r"
GOAL  = G = "g"

U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]
U_MAZE_EVAL = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]
BIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, 1, 1, G, G, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, 1, G, G, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, G, 1, G, G, 1, G, 1],
    [1, G, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]
BIG_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, G, G, 1],
    [1, 0, 0, 1, 0, G, G, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, G, 0, 1, G, 1],
    [1, 0, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]
HARDEST_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, 1, G, G, G, G, G, 1],
    [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
    [1, G, G, G, G, G, G, 1, G, G, G, 1],
    [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
    [1, G, G, 1, G, 1, G, G, G, G, G, 1],
    [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

MAZE_HEIGHT = 0.5
_BASE_XML = Path(__file__).parent / "assets" / "ant_maze.xml"

MAZE_LAYOUTS = {
    "u_maze": U_MAZE,
    "u_maze_eval": U_MAZE_EVAL,
    "big_maze": BIG_MAZE,
    "big_maze_eval": BIG_MAZE_EVAL,
    "hardest_maze": HARDEST_MAZE,
}


def _find_starts(layout, scale):
    return jp.array([[i * scale, j * scale]
                     for i, row in enumerate(layout)
                     for j, cell in enumerate(row) if cell == RESET])


def _find_goals(layout, scale):
    return jp.array([[i * scale, j * scale]
                     for i, row in enumerate(layout)
                     for j, cell in enumerate(row) if cell == GOAL])


def _make_maze_xml(maze_layout_name: str, maze_size_scaling: float):
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


# ── Config ────────────────────────────────────────────────────────────────────
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        impl="warp",
        naconmax=100_000,   # Higher than plain ant: many wall contacts
        njmax=150,
    )


class AntMaze(mjx_env.MjxEnv):
    """AntMaze — sparse reward, goal-conditioned."""

    def __init__(
        self,
        maze_layout_name: str = "u_maze",
        maze_size_scaling: float = 4.0,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        xml_string, possible_starts, possible_goals = _make_maze_xml(
            maze_layout_name, maze_size_scaling
        )
        self.possible_starts = possible_starts
        self.possible_goals  = possible_goals

        self._xml_path = f"ant_maze_{maze_layout_name}"   # logical name only
        self._mj_model = mujoco.MjModel.from_xml_string(xml_string)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        self._target_qpos_ndof = 2
        self._target_qvel_ndof = 2
        self._torso_body_id = 1
        self._healthy_z_min = 0.2
        self._healthy_z_max = 1.0
        self.state_dim = 29
        self.goal_indices = jp.array([0, 1])
        self.goal_reach_thresh = 0.5

    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        noise = 0.1
        qpos = jp.array(self._mj_model.qpos0)
        qpos = qpos + noise * jax.random.uniform(
            rng1, (self._mj_model.nq,), minval=-1.0, maxval=1.0
        )
        qvel = noise * jax.random.normal(rng2, (self._mj_model.nv,))

        start  = self._random_start(rng3)
        target = self._random_target(rng4)
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
        qpos = data.qpos[: -self._target_qpos_ndof]
        qvel = data.qvel[: -self._target_qvel_ndof]
        target_pos = data.qpos[-self._target_qpos_ndof:]
        return jp.concatenate([qpos, qvel, target_pos])

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


# ── Convenience subclasses (match jaxgcrl naming) ────────────────────────────

class AntUMaze(AntMaze):
    def __init__(self, **kwargs):
        super().__init__(maze_layout_name="u_maze", **kwargs)

class AntBigMaze(AntMaze):
    def __init__(self, **kwargs):
        super().__init__(maze_layout_name="big_maze", **kwargs)

class AntHardestMaze(AntMaze):
    def __init__(self, **kwargs):
        super().__init__(maze_layout_name="hardest_maze", **kwargs)