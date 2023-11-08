import multiprocessing
from typing import Any, Dict, Tuple, Union

import dm_env
import gym
import numpy as np
import pygame
import torch
from dm_control import suite

import envs.base_env as base_env

# Define the screen dimensions
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 360


class DMEnv(base_env.BaseEnv):
    """
    Represents an environment using DeepMind Control Suite.

    This environment allows interaction with various physics-based tasks.
    """

    def __init__(self, config: Dict[str, Any], device: Union[str, torch.device], visualize: bool):
        """
        Initialize the DM environment with configuration, device, and visualization settings.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing environment settings.
            device (Union[str, torch.device]): Device to which tensors should be moved.
            visualize (bool): Whether to visualize the environment.
        """
        super().__init__(visualize)
        self._device = device
        self._visualize = visualize
        self._time_limit = config["time_limit"]
        self._env = self._build_dm_env(config)

        if self._visualize:
            self._init_render()

        self.reset()

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """
        Resets the environment to its initial state.

        Returns:
            Tuple[torch.Tensor, Dict]: Initial observation and an empty info dictionary.
        """
        time_step = self._env.reset()
        obs = self._convert_obs(time_step.observation)
        return obs, {}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Executes one time step within the environment.

        Args:
            action (torch.Tensor): The action to perform.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
                Observation, reward, done flag, and an empty info dictionary.
        """
        time_step = self._env.step(action.cpu().numpy())
        obs = self._convert_obs(time_step.observation)
        reward = 0 if time_step.reward is None else time_step.reward
        reward = torch.tensor([reward], device=self._device)
        done = self._compute_done(time_step.last())
        info = {}

        if self._visualize:
            self._render()

        return obs, reward, done, info

    def _build_dm_env(self, config: Dict[str, Any]) -> dm_env.Environment:
        """
        Builds the DeepMind environment based on the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing environment settings.

        Returns:
            dm_env.Environment: The constructed DeepMind environment.
        """
        env_name = config["env_name"]
        task_name = config["task"]
        assert env_name.startswith("dm_")
        env_name = env_name[3:]
        return suite.load(env_name, task_name)

    def _convert_obs(self, obs: Dict[str, Union[int, np.ndarray]]) -> torch.Tensor:
        """
        Converts observations from the environment into a flat torch tensor.

        Args:
            obs (Dict[str, Union[int, np.ndarray]]): A dictionary containing the observed values from the environment.

        Returns:
            torch.Tensor: A tensor representing the flattened observations.
        """
        obs_list = []
        for v in obs.values():
            if not isinstance(v, np.ndarray):
                obs_list.append(np.array([v]))
            else:
                obs_list.append(v)

        flat_obs = np.concatenate(obs_list, axis=-1)
        flat_obs = flat_obs.astype(np.float32)
        flat_obs = torch.from_numpy(flat_obs)
        flat_obs = flat_obs.to(device=self._device)

        return flat_obs

    def _compute_done(self, last: bool) -> torch.Tensor:
        """
        Computes the done state based on the time and whether the last action terminated the environment.

        Args:
            last (bool): A flag indicating whether the last action terminated the environment.

        Returns:
            torch.Tensor: A tensor containing the computed done state.
        """
        if self._env._physics.time() >= self._time_limit:
            done = torch.tensor([base_env.DoneFlags.TIME.value], device=self._device)
        elif last:
            done = torch.tensor([base_env.DoneFlags.FAIL.value], device=self._device)
        else:
            done = torch.tensor([base_env.DoneFlags.NULL.value], device=self._device)
        return done

    def get_action_space(self) -> gym.Space:
        """
        Returns the action space of the environment.

        Returns:
            gym.Space: The action space of the environment.
        """
        action_spec = self._env.action_spec()
        if isinstance(action_spec, dm_env.specs.BoundedArray):
            low = action_spec.minimum
            high = action_spec.maximum
            action_space = gym.spaces.Box(low=low, high=high, shape=low.shape)
        else:
            assert False, "Unsupported DM action type"

        return action_space

    def get_reward_bounds(self) -> Tuple[float, float]:
        """
        Returns the lower and upper bounds of possible rewards in the environment.

        Returns:
            Tuple[float, float]: The lower and upper bounds of possible rewards.
        """
        return 0.0, 1.0

    def _init_render(self) -> None:
        """
        Initializes the rendering process and queue for the environment.
        """
        self._clock = pygame.time.Clock()

        self._render_queue = multiprocessing.Queue()
        self._render_proc = multiprocessing.Process(target=render_worker, args=(SCREEN_WIDTH, SCREEN_HEIGHT, self._render_queue))
        self._render_proc.start()

    def _render(self) -> None:
        """
        Renders the environment state and regulates the rendering speed.
        """
        im = self._env.physics.render(SCREEN_HEIGHT, SCREEN_WIDTH, camera_id=0)
        im = np.transpose(im, axes=[1, 0, 2])
        self._render_queue.put(im)

        fps = 1.0 / self._env.control_timestep()
        self._clock.tick(fps)
        return


def render_worker(screen_w: int, screen_h: int, render_queue: multiprocessing.Queue) -> None:
    """
    Worker function to handle rendering tasks.

    Args:
        screen_w (int): Screen width.
        screen_h (int): Screen height.
        render_queue (multiprocessing.Queue): Queue used to receive rendering tasks.
    """
    pygame.init()
    window = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("DeepMind Control Suite")

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        im = render_queue.get()
        surface = pygame.pixelcopy.make_surface(im)
        window.blit(surface, (0, 0))
        pygame.display.update()

    pygame.quit()
