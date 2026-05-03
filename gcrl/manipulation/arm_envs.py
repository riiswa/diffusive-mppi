# mujoco_playground_envs/manipulation/arm_envs.py
"""Base class for all arm environments."""
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,       # n_frames=25 * sim_dt=0.002
        sim_dt=0.002,
        episode_length=150,
        action_repeat=1,
        vision=False,
        impl="warp",
        naconmax=100_000,
        njmax=200,
    )


class ArmEnvs(mjx_env.MjxEnv):
    """
    MJX-Warp port of jaxgcrl ArmEnvs.

    Key mappings vs Brax:
      pipeline_state.q[i]          → data.qpos[i]
      pipeline_state.x.pos[i]      → data.xpos[i]          (body position)
      pipeline_state.xd.vel[i]     → data.cvel[i, 3:]      (linear velocity, global)
      pipeline_state.qfrc_actuator → data.qfrc_actuator

    Goal storage: state.info["goal"]   (unchanged from jaxgcrl)
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        self._set_environment_attributes()

        xml_path = self._get_xml_path()
        self._xml_path = xml_path
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        # Override sim settings (mirrors jaxgcrl arm_envs)
        self._mj_model.opt.timestep  = self.sim_dt
        self._mj_model.opt.iterations    = 6
        self._mj_model.opt.ls_iterations = 12
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    # ------------------------------------------------------------------
    # Interface to be implemented by subclasses
    # ------------------------------------------------------------------
    def _get_xml_path(self) -> str:
        raise NotImplementedError

    def _set_environment_attributes(self) -> None:
        """Set env_name, episode_length, goal_indices, state_dim, noise scales, etc."""
        raise NotImplementedError

    def _get_initial_state(self, rng: jax.Array):
        """Return (qpos, qvel) as JAX arrays."""
        raise NotImplementedError

    def _get_initial_goal(self, data: mjx.Data, rng: jax.Array) -> jax.Array:
        raise NotImplementedError

    def _compute_goal_completion(self, obs: jax.Array, goal: jax.Array):
        """Return (success, success_easy, success_hard)."""
        raise NotImplementedError

    def _update_goal_visualization(self, data: mjx.Data, goal: jax.Array) -> mjx.Data:
        raise NotImplementedError

    def _get_obs(self, data: mjx.Data, goal: jax.Array, timestep: float) -> jax.Array:
        raise NotImplementedError

    def _get_arm_angles(self, data: mjx.Data) -> jax.Array:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Core environment loop
    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key_init, key_goal, key_post = jax.random.split(rng, 4)

        qpos, qvel = self._get_initial_state(key_init)
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = data.replace(qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        goal = self._get_initial_goal(data, key_goal)
        data = self._update_goal_visualization(data, goal)

        timestep = 0.0
        obs = self._get_obs(data, goal, timestep)

        info = {
            "goal": goal,
            "timestep": 0.0,
            "postexplore_timestep": jax.random.uniform(key_post),
            "rng": rng,
        }
        reward, done, zero = jp.zeros(3)
        metrics = {"success": zero, "success_easy": zero, "success_hard": zero}
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Convert raw action to actuator inputs
        if "EEF" in self.env_name:
            converted = self._convert_action_eef(state.data, action)
        else:
            arm_angles = self._get_arm_angles(state.data)
            converted = self._convert_action_joint_angle(action, arm_angles)

        data = mjx_env.step(self.mjx_model, state.data, converted, self.n_substeps)

        timestep = state.info["timestep"] + 1.0 / self.episode_length
        goal = state.info["goal"]
        obs = self._get_obs(data, goal, timestep)

        success, success_easy, success_hard = self._compute_goal_completion(obs, goal)
        state.metrics.update(
            success=success, success_easy=success_easy, success_hard=success_hard
        )
        reward = success
        done   = jp.zeros(())
        info   = {**state.info, "timestep": timestep}

        new_state = mjx_env.State(data, obs, reward, done, state.metrics, info)

        # arm_grasp: update finger goal to track cube
        if self.env_name == "arm_grasp":
            cube_pos = obs[:3]
            adjusted_goal = goal.at[:3].set(cube_pos + jp.array([0.0375, 0.0, 0.0]))
            adjusted_goal = adjusted_goal.at[3:6].set(cube_pos + jp.array([-0.0375, 0.0, 0.0]))
            new_info = {**new_state.info, "goal": adjusted_goal}
            new_data = self._update_goal_visualization(new_state.data, adjusted_goal)
            new_state = mjx_env.State(new_data, new_state.obs, new_state.reward,
                                      new_state.done, new_state.metrics, new_info)
        return new_state

    def update_goal(self, state: mjx_env.State, goal: jax.Array) -> mjx_env.State:
        """Allow external goal updates during trajectory collection."""
        info = {**state.info, "goal": goal}
        data = self._update_goal_visualization(state.data, goal)
        return mjx_env.State(data, state.obs, state.reward, state.done, state.metrics, info)

    # ------------------------------------------------------------------
    # Action conversion (identical logic to jaxgcrl, MJX-adapted)
    # ------------------------------------------------------------------
    def _convert_action_joint_angle(
        self, action: jax.Array, arm_angles: jax.Array, delta_control: bool = False
    ) -> jax.Array:
        arm_action = jp.array([action[0], action[1], 0.0, action[2], 0.0, action[3], 0.0])
        min_value = jp.array([0.3491,  0.0,     0.0, -3.0718, 0.0, 2.3562, 1.4487])
        max_value = jp.array([2.7925,  1.48353, 0.0, -0.0698, 0.0, 3.7525, 1.4487])
        offset     = (min_value + max_value) / 2.0
        multiplier = (max_value - min_value) / 2.0

        if delta_control:
            norm_angles = jp.where(multiplier > 0, (arm_angles - offset) / multiplier, 0.0)
            arm_action  = jp.clip(norm_angles + arm_action * 0.25, -1.0, 1.0)

        arm_action = offset + arm_action * multiplier

        if self.env_name not in ("arm_reach",):
            gripper = jp.where(
                action[-1] > 0,
                jp.array([0.0, 0.0]),
                jp.array([255.0, 255.0]),
            )
            return jp.concatenate([arm_action, gripper])
        return arm_action

    def _convert_action_eef(
        self, data: mjx.Data, action: jax.Array
    ) -> jax.Array:
        eef_index = 2
        current_pos = data.xpos[eef_index]
        arm_action  = current_pos + 0.2 * jp.clip(action[:3], -1.0, 1.0)
        gripper = jp.where(
            action[-1] > 0,
            jp.array([0.0, 0.0]),
            jp.array([255.0, 255.0]),
        )
        return jp.concatenate([arm_action, gripper])

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