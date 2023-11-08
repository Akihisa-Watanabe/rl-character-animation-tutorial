from typing import Any, Tuple, Union

import gym
import numpy as np
import torch

import envs.atari_wrappers as atari_wrappers
import envs.base_env as base_env


class AtariEnv(base_env.BaseEnv):
    def __init__(self, config: dict, device: torch.device, visualize: bool) -> None:
        """
        Initializes the Atari environment.

        Args:
            config (dict): Configuration dictionary specifying environment settings.
            device (torch.device): Device to perform computations on.
            visualize (bool): Whether to visualize the environment or not.
        """
        super().__init__(visualize)
        self._device = device
        self._env = self._build_atari_env(config, visualize)
        self.reset()

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """
        Resets the environment.

        Returns:
            Tuple[torch.Tensor, Any]: The initial observation and info from the environment.
        """
        obs, info = self._env.reset()
        self._curr_obs = self._convert_obs(obs)
        self._curr_info = info
        return self._curr_obs, self._curr_info

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        Takes a step in the environment using the given action.

        Args:
            action (torch.Tensor): Action to be taken in the environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
            Observation, reward, done flag, and info from the environment.
        """
        a = action.cpu().numpy()[0]
        obs, reward, terminated, truncated, info = self._env.step(a)
        self._curr_obs = self._convert_obs(obs)
        self._curr_info = info

        reward = self._handle_reward(reward)
        done = self._compute_done(terminated, truncated)

        return self._curr_obs, reward, done, self._curr_info

    def _build_atari_env(self, config: dict, visualize: bool) -> gym.Env:
        """
        Constructs the Atari environment based on the given configuration.

        Args:
            config (dict): Configuration dictionary specifying environment settings.
            visualize (bool): Whether to visualize the environment or not.

        Returns:
            gym.Env: The constructed Atari environment.
        """
        env_name = config.get("env_name", "")
        assert env_name.startswith("atari_"), "Invalid environment name"

        env_name = env_name[6:] + "NoFrameskip-v4"
        max_timesteps = config.get("max_timesteps", None)

        atari_env = atari_wrappers.make_atari(env_name, max_episode_steps=max_timesteps, visualize=visualize)
        return atari_wrappers.wrap_deepmind(atari_env, warp_frame=True, frame_stack=True)

    def _convert_obs(self, obs: np.ndarray) -> torch.Tensor:
        """
        Converts the observation to a torch tensor.

        Args:
            obs (np.ndarray): The observation from the environment.

        Returns:
            torch.Tensor: The observation as a torch tensor.
        """
        frames = [np.expand_dims(obs.frame(i), axis=-3) for i in range(obs.count())]
        frames = np.concatenate(frames, axis=-3)
        obs = torch.tensor(frames, device=self._device).unsqueeze(0)
        return obs

    def _handle_reward(self, reward: Union[float, Tuple[float, float]]) -> torch.Tensor:
        """
        Handles the received reward based on the current mode (train or test).

        Args:
            reward (Union[float, Tuple[float, float]]): The received reward.

        Returns:
            torch.Tensor: The handled reward as a torch tensor.
        """
        if self._mode is base_env.EnvMode.TRAIN:
            reward = reward[1]  # clipped reward
        else:
            reward = reward[0]  # unclipped reward

        return torch.tensor([reward], device=self._device)

    def _compute_done(self, terminated: bool, truncated: bool) -> torch.Tensor:
        """
        Computes the 'done' flag based on whether the episode was terminated or truncated.

        Args:
            terminated (bool): Whether the episode was terminated.
            truncated (bool): Whether the episode was truncated.

        Returns:
            torch.Tensor: The 'done' flag as a torch tensor.
        """
        if terminated:
            return torch.tensor([base_env.DoneFlags.FAIL.value], device=self._device)
        elif truncated:
            return torch.tensor([base_env.DoneFlags.TIME.value], device=self._device)
        else:
            return torch.tensor([base_env.DoneFlags.NULL.value], device=self._device)

    def get_obs_space(self) -> gym.spaces.Box:
        """
        Gets the observation space of the environment.

        Returns:
            gym.spaces.Box: The observation space of the environment.
        """
        obs_space = self._env.observation_space
        obs_shape = obs_space.shape[-1:] + obs_space.shape[:-1]
        return gym.spaces.Box(low=obs_space.low.min(), high=obs_space.high.max(), shape=obs_shape, dtype=obs_space.dtype)

    def get_action_space(self) -> gym.Space:
        """
        Gets the action space of the environment.

        Returns:
            gym.Space: The action space of the environment.
        """
        a_space = self._env.action_space
        a_space._shape = (1,)
        return a_space

    def get_reward_bounds(self) -> Tuple[float, float]:
        """
        Gets the reward bounds of the environment.

        Returns:
            Tuple[float, float]: The reward bounds of the environment.
        """
        return -1.0, 1.0
