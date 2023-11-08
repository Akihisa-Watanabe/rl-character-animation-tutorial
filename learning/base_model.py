import gym
import numpy as np
import torch

import learning.distribution_categorical as distribution_categorical
import learning.distribution_gaussian_diag as distribution_gaussian_diag
import util.torch_util as torch_util


class BaseModel(torch.nn.Module):
    """
    BaseModel is a foundational class that constructs an action distribution
    based on the action space of the provided environment. It's designed
    to be extended by specific models for Reinforcement Learning algorithms.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        env (gym.Env): The Gym environment.
    """

    def __init__(self, config, env):
        super().__init__()
        self._activation = torch.nn.ReLU

    def _build_action_distribution(self, config, env, input):
        """
        Constructs an action distribution model based on the environment's action space.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            env (gym.Env): The Gym environment.
            input (torch.Tensor): The input tensor. Typically, this would be the state.

        Returns:
            torch.nn.Module: A PyTorch Module representing the action distribution.

        Raises:
            AssertionError: If the action space type is unsupported.
        """
        # Determine the type of action space provided by the environment
        a_space = env.get_action_space()

        # Calculate the output size for the input tensor (state)
        in_size = torch_util.calc_layers_out_size(input)

        # If the action space is continuous (Box)
        if isinstance(a_space, gym.spaces.Box):
            a_size = np.prod(a_space.shape)
            a_init_output_scale = config["actor_init_output_scale"]
            a_std_type = distribution_gaussian_diag.StdType[config["actor_std_type"]]
            a_std = config["action_std"]

            # Create Gaussian Distribution for action selection
            a_dist = distribution_gaussian_diag.DistributionGaussianDiagBuilder(in_size, a_size, std_type=a_std_type, init_std=a_std, init_output_scale=a_init_output_scale)

        # If the action space is discrete (Discrete)
        elif isinstance(a_space, gym.spaces.Discrete):
            num_actions = a_space.n
            a_init_output_scale = config["actor_init_output_scale"]

            # Create Categorical Distribution for action selection
            a_dist = distribution_categorical.DistributionCategoricalBuilder(in_size, num_actions, init_output_scale=a_init_output_scale)

        else:
            raise AssertionError(f"Unsupported action space: {a_space}")

        return a_dist
