import enum

import numpy as np
import torch


class DistributionCategoricalBuilder(torch.nn.Module):
    """
    A PyTorch Module that builds a neural network for generating logits
    for actions in a categorical distribution.

    Args:
        in_size (int): Dimensionality of the input state space.
        out_size (int): Number of possible actions.
        init_output_scale (float, optional): Initial scale for output layer weights.
                                             Defaults to 0.01.
    """

    def __init__(self, in_size: int, out_size: int, init_output_scale: float = 0.01) -> None:
        super().__init__()
        self._build_params(in_size, out_size, init_output_scale)

    def _build_params(self, in_size: int, out_size: int, init_output_scale: float) -> None:
        """
        Internal method for initializing the parameters of the network.

        Args:
            in_size (int): Dimensionality of the input state space.
            out_size (int): Number of possible actions.
            init_output_scale (float): Initial scale for output layer weights.
        """
        self._logit_net = torch.nn.Linear(in_size, out_size)
        torch.nn.init.uniform_(self._logit_net.weight, -init_output_scale, init_output_scale)

    def forward(self, input: torch.Tensor) -> "DistributionCategorical":
        """
        Forward pass for the network. Computes the logits for the given input state.

        Args:
            input (torch.Tensor): The input state.

        Returns:
            DistributionCategorical: A categorical distribution parameterized by the logits.
        """
        logits = self._logit_net(input)
        dist = DistributionCategorical(logits=logits)
        return dist


class DistributionCategorical(torch.distributions.Categorical):
    """
    A wrapper around PyTorch's Categorical distribution class with custom methods.

    Args:
        logits (torch.Tensor): The logits tensor.
    """

    def __init__(self, logits: torch.Tensor) -> None:
        logits = logits.unsqueeze(-2)
        super().__init__(logits=logits)

    @property
    def mode(self) -> torch.Tensor:
        """
        Returns the action with the highest logit value (the most probable action).

        Returns:
            torch.Tensor: The action with the highest logit value.
        """
        arg_max = torch.argmax(self.logits, dim=-1)
        return arg_max

    def sample(self) -> torch.Tensor:
        """
        Samples an action based on the logits.

        Returns:
            torch.Tensor: The sampled action.
        """
        x = super().sample()
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the log probability of a given action.

        Args:
            x (torch.Tensor): The action for which to compute the log probability.

        Returns:
            torch.Tensor: The log probability of the action.
        """
        logp = super().log_prob(x)
        logp = logp.squeeze(-1)
        return logp

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of the action distribution.

        Returns:
            torch.Tensor: The entropy of the action distribution.
        """
        ent = super().entropy()
        ent = ent.squeeze(-1)
        return ent

    def param_reg(self) -> torch.Tensor:
        """
        Computes the regularization term based on the logits.

        Returns:
            torch.Tensor: The regularization term.
        """
        # only regularize mean, covariance is regularized via entropy reg
        reg = torch.sum(torch.square(self.logits), dim=-1)
        reg = reg.squeeze(-1)
        return reg
