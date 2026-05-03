# mujoco_playground_envs/manipulation/arm_grasp.py
from pathlib import Path

import jax
import jax.numpy as jp
from mujoco import mjx

from .arm_envs import ArmEnvs


class ArmGrasp(ArmEnvs):
    """Grasp: close fingers on opposite sides of a cube."""

    def _get_xml_path(self) -> str:
        return str(Path(__file__).parent.parent / "assets" / "panda_grasp.xml")

    @property
    def action_size(self) -> int:
        return 5

    def _set_environment_attributes(self) -> None:
        self.env_name = "arm_grasp"
        self.episode_length = 100
        # Goal: [left_fingertip_pos(3), right_fingertip_pos(3), finger_distance(1)]
        self.goal_indices = jp.array([16, 17, 18, 19, 20, 21, 22])
        self.completion_goal_indices = jp.array([16, 17, 18, 19, 20, 21, 22])
        self.state_dim = 24
        self.arm_noise_scale  = 0.0
        self.cube_noise_scale = 0.3

    def _get_initial_state(self, rng):
        rng, key1, key2 = jax.random.split(rng, 3)
        cube_q_xy  = jp.array(self._mj_model.qpos0[:2]) + self.cube_noise_scale * jax.random.uniform(key1, [2], minval=-1.0)
        cube_rest  = jp.array(self._mj_model.qpos0[2:7])
        target_q   = jp.array(self._mj_model.qpos0[7:14])
        arm_default = jp.array([1.571, 0.742, 0.0, -1.571, 0.0, 3.054, 1.449, 0.04, 0.04, 0.0, 0.0])
        arm_q = arm_default + 0.0 * jax.random.uniform(key2, [self._mj_model.nq - 14], minval=-1.0)
        qpos = jp.concatenate([cube_q_xy, cube_rest, target_q, arm_q])
        qvel = jp.zeros(self._mj_model.nv)
        return qpos, qvel

    def _get_initial_goal(self, data: mjx.Data, rng) -> jax.Array:
        cube_pos = data.qpos[:3]
        return jp.concatenate([
            cube_pos + jp.array([0.0375, 0.0, 0.0]),
            cube_pos + jp.array([-0.0375, 0.0, 0.0]),
            jp.array([0.075]),
        ])

    def _compute_goal_completion(self, obs, goal):
        cube_pos  = obs[:3]
        left_tip  = obs[16:19]
        right_tip = obs[19:22]
        midpoint  = (left_tip + right_tip) / 2.0
        dist_mid  = jp.linalg.norm(cube_pos - midpoint)
        openness  = obs[22]
        goal_open = goal[9] if goal.shape[0] > 9 else jp.array(0.075)
        open_diff = jp.abs(openness - goal_open)

        success = jp.array(jp.all(jp.array([dist_mid < 0.05, open_diff < 0.02])), dtype=float)
        success_easy = jp.array(jp.all(jp.array([dist_mid < 0.15, open_diff < 0.05])), dtype=float)
        success_hard = jp.array(jp.all(jp.array([dist_mid < 0.02, open_diff < 0.005])), dtype=float)
        return success, success_easy, success_hard

    def _update_goal_visualization(self, data: mjx.Data, goal: jax.Array) -> mjx.Data:
        new_qpos = data.qpos.at[7:10].set(goal[:3])
        return data.replace(qpos=new_qpos)

    def _get_obs(self, data: mjx.Data, goal: jax.Array, timestep) -> jax.Array:
        """24-dim obs: q_subset(10) + EEF(6) + fingertips(6) + distance(1) + force(1)."""
        q_idx    = jp.array([0, 1, 2, 14, 15, 16, 17, 18, 19, 20])
        q_subset = data.qpos[q_idx]

        eef_index = 8
        eef_pos = data.xpos[eef_index]
        eef_vel = data.cvel[eef_index, 3:]

        left_tip_idx  = 10
        right_tip_idx = 12
        left_tip_pos  = data.xpos[left_tip_idx]
        right_tip_pos = data.xpos[right_tip_idx]
        tip_dist      = jp.linalg.norm(right_tip_pos - left_tip_pos)[None]

        finger_force_idx = jp.array([-4, -2])
        gripper_force = data.qfrc_actuator[finger_force_idx].mean(keepdims=True) * 0.1

        return jp.concatenate([
            q_subset, eef_pos, eef_vel,
            left_tip_pos, right_tip_pos, tip_dist,
            gripper_force, goal,
        ])

    def _get_arm_angles(self, data: mjx.Data) -> jax.Array:
        return data.qpos[14:21]