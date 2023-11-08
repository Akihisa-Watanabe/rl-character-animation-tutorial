import abc
import enum
from typing import Any, List, Optional, Tuple, Union

import gym
import numpy as np

import util.torch_util as torch_util


class EnvMode(enum.Enum):
    """Enumeration for environment modes."""

    TRAIN = 0
    TEST = 1


class DoneFlags(enum.Enum):
    """Enumeration for termination flags of the environment."""

    NULL = 0
    FAIL = 1
    SUCC = 2
    TIME = 3


class BaseEnv(abc.ABC):
    """
    Abstract base class for an environment.

    Attributes:
        NAME (str): Name of the environment.
    """

    NAME: str = "base"

    def __init__(self, visualize: bool) -> None:
        """
        Initializes the environment.

        Args:
            visualize (bool): Whether to visualize the environment.
        """
        self._mode: EnvMode = EnvMode.TRAIN
        self._visualize: bool = visualize
        self._action_space: Optional[gym.Space] = None

    @abc.abstractmethod
    def reset(self, env_ids: Optional[Any] = None) -> Any:
        """
        Resets the environment state.

        Args:
            env_ids (Optional[Any]): Environment IDs to reset. Default is None.

        Returns:
            Any: Initial observation of the environment.
        """
        pass

    @abc.abstractmethod
    def step(self, action: Any) -> Any:
        """
        Takes an action in the environment.

        Args:
            action (Any): The action to be taken.

        Returns:
            Any: Observation, reward, done flag, and additional info.
        """
        pass

    def get_obs_space(self) -> gym.Space:
        """
        Returns the observation space of the environment.

        Returns:
            gym.Space: The observation space.
        """
        obs, _ = self.reset()
        obs_shape: List[int] = list(obs.shape)
        obs_dtype = torch_util.torch_dtype_to_numpy(obs.dtype)
        obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=obs_dtype,
        )
        return obs_space

    def get_action_space(self) -> Optional[gym.Space]:
        """
        Returns the action space of the environment.

        Returns:
            Optional[gym.Space]: The action space.
        """
        return self._action_space

    def set_mode(self, mode: EnvMode) -> None:
        """
        Sets the environment mode.

        Args:
            mode (EnvMode): The environment mode.
        """
        self._mode = mode

    def get_num_envs(self) -> int:
        """
        Returns the number of environments.

        Returns:
            int: The number of environments (always 1 in this base class).
        """
        return 1

    def get_reward_bounds(self) -> Tuple[float, float]:
        """
        Returns the minimum and maximum possible rewards.

        Returns:
            Tuple[float, float]: The minimum and maximum rewards.
        """
        return (-np.inf, np.inf)

    def get_reward_fail(self) -> float:
        """
        Returns the reward for failing an episode.

        Returns:
            float: The reward for failure (always 0 in this base class).
        """
        return 0.0

    def get_reward_succ(self) -> float:
        """
        Returns the reward for successfully completing an episode.

        Returns:
            float: The reward for success (always 0 in this base class).
        """
        return 0.0

    def get_visualize(self) -> bool:
        """
        Returns whether the environment should be visualized or not.

        Returns:
            bool: True if the environment should be visualized, False otherwise.
        """
        return self._visualize
