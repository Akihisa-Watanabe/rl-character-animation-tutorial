import math
import time
from typing import Any, Tuple, Union

import gym
import numpy as np
import torch

import envs.base_env as base_env

# Define the screen dimensions
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 360


class ClassicEnv(base_env.BaseEnv):
    def __init__(self, config: dict, device: torch.device, visualize: bool) -> None:
        """
        Initializes the Classic Controle environment.

        Args:
            config (dict): Configuration dictionary specifying environment settings.
            device (torch.device): Device to perform computations on.
            visualize (bool): Whether to visualize the environment or not.
        """
        super().__init__(visualize)
        self._device = device
        self._env = self._build_classic_env(config, visualize)
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
        # Convert the action to a valid format based on the action space type
        if isinstance(self._env.action_space, gym.spaces.Discrete):
            a = action.item()  # For discrete action space
        elif isinstance(self._env.action_space, gym.spaces.Box):
            a = action.cpu().numpy()  # For continuous action space
        else:
            raise NotImplementedError(f"Action space type {type(self._env.action_space)} not supported")

        obs, reward, terminated, truncated, info = self._env.step(a)
        self._curr_obs = self._convert_obs(obs)
        self._curr_info = info

        reward = self._handle_reward(reward)
        done = self._compute_done(terminated, truncated)

        if self._visualize:
            self._env.render()
            time.sleep(0.1)

        return self._curr_obs, reward, done, self._curr_info

    def _build_classic_env(self, config: dict, visualize: bool) -> gym.Env:
        """
        Builds the Classic Control environment.

        Args:
            config (dict): Configuration dictionary specifying environment settings.
            visualize (bool): Whether to visualize the environment or not.

        Returns:
            gym.Env: The constructed environment.
        """
        env_name = config.get("env_name", "")
        assert env_name.startswith("classic_"), "Invalid environment name"

        env_name = env_name[8:]
        if self._visualize:
            env = gym.make(env_name, render_mode="human")
            if env_name == "cartpole":
                env.theta_threshold_radians = 180 * math.pi / 180
        else:
            env = gym.make(env_name)
            # if env_name == "cartpole":
            # env.theta_threshold_radians = 30 * math.pi / 180
        return env

    def _convert_obs(self, obs: np.ndarray) -> torch.Tensor:
        """
        Converts the observation to a torch tensor.

        Args:
            obs (np.ndarray): The observation from the environment.

        Returns:
            torch.Tensor: The observation as a torch tensor.
        """
        obs = np.expand_dims(obs, axis=0)
        obs = torch.tensor(obs, device=self._device).unsqueeze(0)
        return obs

    def _handle_reward(self, reward: Union[float, Tuple[float, float]]) -> torch.Tensor:
        """
        Handles the received reward based on the current mode (train or test).

        Args:
            reward (Union[float, Tuple[float, float]]): The received reward.

        Returns:
            torch.Tensor: The handled reward as a torch tensor.
        """
        if isinstance(reward, float) or isinstance(reward, int):  # Handle scalar reward
            return torch.tensor([reward], device=self._device)
        elif isinstance(reward, tuple):  # Handle tuple reward
            if self._mode is base_env.EnvMode.TRAIN:
                return torch.tensor([reward[1]], device=self._device)  # clipped reward
            else:
                return torch.tensor([reward[0]], device=self._device)  # unclipped reward
        else:
            raise TypeError(f"Unsupported reward type: {type(reward)}")

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
