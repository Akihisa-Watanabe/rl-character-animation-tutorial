import enum
from typing import Union

import numpy as np
import torch


# Define types of standard deviation (std) for Gaussian distribution
class StdType(enum.Enum):
    FIXED = 0  # The standard deviation is fixed and not trainable.
    CONSTANT = 1  # The standard deviation is constant but trainable.
    VARIABLE = 2  # The standard deviation is a function of the input and trainable.


# Neural network-based builder for Gaussian distributions with diagonal covariance
class DistributionGaussianDiagBuilder(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int, std_type: StdType, init_std: float, init_output_scale: float = 0.01) -> None:
        """
        Initialize the distribution builder.

        Args:
            in_size (int): Dimensionality of the input.
            out_size (int): Dimensionality of the output (mean).
            std_type (StdType): Type of standard deviation.
            init_std (float): Initial value for standard deviation.
            init_output_scale (float): Initial scale for output weights.
        """
        super().__init__()
        self._std_type = std_type

        self._build_params(in_size, out_size, init_std, init_output_scale)

    def _build_params(self, in_size: int, out_size: int, init_std: float, init_output_scale: float) -> None:
        """
        Build the internal parameters.
        """
        # Build mean network
        self._mean_net = torch.nn.Linear(in_size, out_size)
        torch.nn.init.uniform_(self._mean_net.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._mean_net.bias)

        # Build log standard deviation network or parameter based on the type
        logstd = np.log(init_std)
        if self._std_type == StdType.FIXED:
            self._logstd_net = torch.nn.Parameter(torch.zeros(out_size, requires_grad=False, dtype=torch.float32), requires_grad=False)
            torch.nn.init.constant_(self._logstd_net, logstd)
        elif self._std_type == StdType.CONSTANT:
            self._logstd_net = torch.nn.Parameter(torch.zeros(out_size, requires_grad=True, dtype=torch.float32), requires_grad=True)
            torch.nn.init.constant_(self._logstd_net, logstd)
        elif self._std_type == StdType.VARIABLE:
            self._logstd_net = torch.nn.Linear(in_size, out_size)
            torch.nn.init.uniform_(self._logstd_net.weight, -init_output_scale, init_output_scale)
            torch.nn.init.constant_(self._logstd_net.bias, logstd)
        else:
            assert False, "Unsupported StdType: {}".format(self._std_type)

        return

    def forward(self, input: torch.Tensor) -> "DistributionGaussianDiag":
        """
        Forward pass to get the Gaussian distribution parameters.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            DistributionGaussianDiag: Gaussian distribution parameterized by the mean and logstd.
        """
        mean = self._mean_net(input)

        if self._std_type == StdType.FIXED or self._std_type == StdType.CONSTANT:
            logstd = torch.broadcast_to(self._logstd_net, mean.shape)
        elif self._std_type == StdType.VARIABLE:
            logstd = self._logstd_net(input)
        else:
            assert False, "Unsupported StdType: {}".format(self._std_type)

        dist = DistributionGaussianDiag(mean=mean, logstd=logstd)
        return dist


class DistributionGaussianDiag:
    def __init__(self, mean: torch.Tensor, logstd: torch.Tensor) -> None:
        """
        Initialize Gaussian distribution with diagonal covariance.

        Args:
            mean (torch.Tensor): Mean of the distribution.
            logstd (torch.Tensor): Logarithm of the standard deviation of the distribution.
        """
        self._mean = mean
        self._logstd = logstd
        self._std = torch.exp(self._logstd)
        self._dim = self._mean.shape[-1]

    @property
    def stddev(self) -> torch.Tensor:
        """Get the standard deviation."""
        return self._std

    @property
    def logstd(self) -> torch.Tensor:
        """Get the logarithm of the standard deviation."""
        return self._logstd

    @property
    def mean(self) -> torch.Tensor:
        """Get the mean."""
        return self._mean

    @property
    def mode(self) -> torch.Tensor:
        """Get the mode, which is the mean for Gaussian distribution."""
        return self._mean

    def sample(self) -> torch.Tensor:
        """Sample from the distribution."""
        noise = torch.normal(torch.zeros_like(self._mean), torch.ones_like(self._std))
        x = self._mean + self._std * noise
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of given sample x."""
        diff = x - self._mean
        logp = -0.5 * torch.sum(torch.square(diff / self._std), dim=-1)
        logp += -0.5 * self._dim * np.log(2.0 * np.pi) - torch.sum(self._logstd, dim=-1)
        return logp

    def entropy(self) -> torch.Tensor:
        """Compute the entropy of the distribution."""
        ent = torch.sum(self._logstd, dim=-1)
        ent += 0.5 * self._dim * np.log(2.0 * np.pi * np.e)
        return ent

    def kl(self, other: "DistributionGaussianDiag") -> torch.Tensor:
        """Compute the KL divergence between this distribution and another."""
        assert isinstance(other, DistributionGaussianDiag)
        other_var = torch.square(other.stddev)
        res = torch.sum(other.logstd - self._logstd + (torch.square(self._std) + torch.square(self._mean - other.mean)) / (2.0 * other_var), dim=-1)
        res += -0.5 * self._dim
        return res

    def param_reg(self):
        # only regularize mean, covariance is regularized via entropy reg
        reg = torch.sum(torch.square(self._mean), dim=-1)
        return reg
