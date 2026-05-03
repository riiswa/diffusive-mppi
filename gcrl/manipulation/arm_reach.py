# mujoco_playground_envs/manipulation/arm_reach.py
from pathlib import Path

import jax
import jax.numpy as jp
from mujoco import mjx

from gcrl.manipulation.arm_envs import ArmEnvs


class ArmReach(ArmEnvs):
    """Reach: move end-effector to a random 3-D goal."""

    def _get_xml_path(self) -> str:
        return str(Path(__file__).parent.parent / "assets" / "panda_reach.xml")

    @property
    def action_size(self) -> int:
        return 4

    def _set_environment_attributes(self) -> None:
        self.env_name = "arm_reach"
        self.episode_length = 100
        self.goal_indices = jp.array([7, 8, 9])         # end-effector position in obs
        self.completion_goal_indices = jp.array([7, 8, 9])
        self.state_dim = 13
        self.goal_reach_thresh = 0.1
        self.arm_noise_scale  = 0.0
        self.goal_noise_scale = 0.2

    def _get_initial_state(self, rng):
        target_q = jp.array(self._mj_model.qpos0[:7])
        arm_default = jp.array([1.571, 0.742, 0.0, -1.571, 0.0, 3.054, 1.449])
        arm_q = arm_default + self.arm_noise_scale * jax.random.uniform(
            rng, [self._mj_model.nq - 7], minval=-1.0
        )
        qpos = jp.concatenate([target_q, arm_q])
        qvel = jp.zeros(self._mj_model.nv)
        return qpos, qvel

    def _get_initial_goal(self, data: mjx.Data, rng) -> jax.Array:
        return jp.array([0.0, 0.5, 0.3]) + self.goal_noise_scale * jax.random.uniform(
            rng, [3], minval=-1.0
        )

    def _compute_goal_completion(self, obs, goal):
        eef_pos  = obs[self.completion_goal_indices]
        dist     = jp.linalg.norm(eef_pos - goal[:3])
        success      = jp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jp.array(dist < 0.3, dtype=float)
        success_hard = jp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard

    def _update_goal_visualization(self, data: mjx.Data, goal: jax.Array) -> mjx.Data:
        new_qpos = data.qpos.at[:3].set(goal)
        return data.replace(qpos=new_qpos)

    def _get_obs(self, data: mjx.Data, goal: jax.Array, timestep) -> jax.Array:
        """13-dim obs: 7-dim joint angles + 6-dim EEF (pos + vel)."""
        q_subset = data.qpos[7:14]
        eef_index = 7    # link 7 = EEF
        eef_pos = data.xpos[eef_index]
        eef_vel = data.cvel[eef_index, 3:]   # linear velocity
        return jp.concatenate([q_subset, eef_pos, eef_vel, goal])

    def _get_arm_angles(self, data: mjx.Data) -> jax.Array:
        return data.qpos[7:14]