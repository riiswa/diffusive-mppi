# mujoco_playground_envs/manipulation/arm_binpick_easy.py
from pathlib import Path

import jax
import jax.numpy as jp
from mujoco import mjx

from .arm_envs import ArmEnvs


def _binpick_obs(data: mjx.Data, goal: jax.Array) -> jax.Array:
    """Shared 18-dim obs for binpick environments (joint-angle control)."""
    q_idx    = jp.array([0, 1, 2, 14, 15, 16, 17, 18, 19, 20])
    q_subset = data.qpos[q_idx]

    eef_index = 8
    eef_pos = data.xpos[eef_index]
    eef_vel = data.cvel[eef_index, 3:]

    left_idx  = 9
    right_idx = 10
    finger_dist = jp.linalg.norm(data.xpos[right_idx] - data.xpos[left_idx])[None]
    gripper_force = data.qfrc_actuator[:-2].mean(keepdims=True) * 0.1

    return jp.concatenate([q_subset, eef_pos, eef_vel, finger_dist, gripper_force, goal])


class ArmBinpickEasy(ArmEnvs):
    """Binpick-Easy: fixed goal location in red bin."""

    def _get_xml_path(self) -> str:
        return str(Path(__file__).parent.parent / "assets" / "panda_binpick_easy.xml")

    @property
    def action_size(self) -> int:
        return 5

    def _set_environment_attributes(self) -> None:
        self.env_name = "arm_binpick_easy"
        self.episode_length = 150
        self.goal_indices = jp.array([0, 1, 2])
        self.completion_goal_indices = jp.array([0, 1, 2])
        self.state_dim = 18
        self.goal_reach_thresh = 0.1
        self.arm_noise_scale  = 0.0
        self.cube_noise_scale = 0.07
        self.goal_noise_scale = 0.005

    def _get_initial_state(self, rng):
        rng, key1, key2 = jax.random.split(rng, 3)
        cube_q_xy  = jp.array(self._mj_model.qpos0[:2]) + self.cube_noise_scale * jax.random.uniform(key1, [2], minval=-1.0)
        cube_rest  = jp.array(self._mj_model.qpos0[2:7])
        target_q   = jp.array(self._mj_model.qpos0[7:14])
        arm_default = jp.array([1.571, 0.742, 0.0, -1.571, 0.0, 3.054, 1.449, 0.04, 0.04])
        qpos = jp.concatenate([cube_q_xy, cube_rest, target_q, arm_default])
        qvel = jp.zeros(self._mj_model.nv)
        return qpos, qvel

    def _get_initial_goal(self, data: mjx.Data, rng) -> jax.Array:
        return jp.array([0.17, 0.6, 0.03]) + jp.array([
            self.goal_noise_scale, self.goal_noise_scale, 0.0
        ]) * jax.random.uniform(rng, [3], minval=-1.0)

    def _compute_goal_completion(self, obs, goal):
        dist = jp.linalg.norm(obs[self.completion_goal_indices] - goal[:3])
        return (
            jp.array(dist < self.goal_reach_thresh, dtype=float),
            jp.array(dist < 0.3, dtype=float),
            jp.array(dist < 0.03, dtype=float),
        )

    def _update_goal_visualization(self, data: mjx.Data, goal: jax.Array) -> mjx.Data:
        return data.replace(qpos=data.qpos.at[7:10].set(goal[:3]))

    def _get_obs(self, data: mjx.Data, goal: jax.Array, timestep) -> jax.Array:
        return _binpick_obs(data, goal)

    def _get_arm_angles(self, data: mjx.Data) -> jax.Array:
        return data.qpos[14:21]