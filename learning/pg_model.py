from typing import Any, Dict

import torch
from gym import Env

import learning.base_model as base_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class PGModel(base_model.BaseModel):
    def __init__(self, config: Dict[str, Any], env: Env) -> None:
        """Initialize the Policy Gradient model.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            env (gym.Env): OpenAI Gym environment.
        """
        super().__init__(config, env)
        self._build_nets(config, env)

    def eval_actor(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        """Evaluate the actor network to get the action distribution.

        Args:
            obs (torch.Tensor): Observations.

        Returns:
            torch.distributions.Distribution: Action distribution.
        """
        h = self._actor_layers(obs)
        a_dist = self._action_dist(h)
        return a_dist

    def eval_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the critic network to get the state value.

        Args:
            obs (torch.Tensor): Observations.

        Returns:
            torch.Tensor: State value.
        """
        h = self._critic_layers(obs)
        val = self._critic_out(h)
        return val

    def _build_nets(self, config: Dict[str, Any], env: Env) -> None:
        """Build both actor and critic networks."""
        self._build_actor(config, env)
        if "critic_net" in config:
            self._build_critic(config, env)

    def _build_actor(self, config: Dict[str, Any], env: Env) -> None:
        """Build the actor network."""
        net_name = config["actor_net"]
        input_dict = self._build_actor_input_dict(env)
        self._actor_layers, layers_info = net_builder.build_net(net_name, input_dict, activation=self._activation)

        self._action_dist = self._build_action_distribution(config, env, self._actor_layers)

    def _build_critic(self, config: Dict[str, Any], env: Env) -> None:
        """Build the critic network."""
        net_name = config["critic_net"]
        input_dict = self._build_critic_input_dict(env)
        self._critic_layers, layers_info = net_builder.build_net(net_name, input_dict, activation=self._activation)
        layers_out_size = torch_util.calc_layers_out_size(self._critic_layers)
        self._critic_out = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.zeros_(self._critic_out.bias)

    def _build_actor_input_dict(self, env: Env) -> Dict[str, Any]:
        """Prepare the actor input dictionary based on the environment's observation space.

        Args:
            env (gym.Env): OpenAI Gym environment.

        Returns:
            dict: Dictionary containing actor input information.
        """
        obs_space = env.get_obs_space()
        input_dict = {"obs": obs_space}
        return input_dict

    def _build_critic_input_dict(self, env: Env) -> Dict[str, Any]:
        """Prepare the critic input dictionary based on the environment's observation space.

        Args:
            env (gym.Env): OpenAI Gym environment.

        Returns:
            dict: Dictionary containing critic input information.
        """
        obs_space = env.get_obs_space()
        input_dict = {"obs": obs_space}
        return input_dict
