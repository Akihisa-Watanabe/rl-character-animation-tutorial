from typing import Optional, Tuple

import numpy as np
import torch


class Normalizer(torch.nn.Module):
    def __init__(self, shape: Tuple[int], device: torch.device, init_mean: Optional[torch.Tensor] = None, init_std: Optional[torch.Tensor] = None, eps: float = 1e-4, clip: float = np.inf, dtype: torch.dtype = torch.float):
        """
        Normalizer class to normalize and unnormalize data tensors.

        Args:
            shape (Tuple[int]): Shape of the tensor to be normalized.
            device (torch.device): The device on which tensors will be allocated.
            init_mean (Optional[torch.Tensor]): Initial mean tensor. Default is None.
            init_std (Optional[torch.Tensor]): Initial standard deviation tensor. Default is None.
            eps (float): Small epsilon value to avoid division by zero. Default is 1e-4.
            clip (float): Value to clip the normalized tensor. Default is np.inf.
            dtype (torch.dtype): Data type of the tensors. Default is torch.float.
        """
        super().__init__()

        self._eps = eps
        self._clip = clip
        self.dtype = dtype
        self._build_params(shape, device, init_mean, init_std)

    def record(self, x: torch.Tensor) -> None:
        """
        Record the given tensor to update the normalizer statistics.

        Args:
            x (torch.Tensor): Tensor to record.
        """
        shape = self.get_shape()
        assert len(x.shape) > len(shape)
        x = x.flatten(start_dim=0, end_dim=len(x.shape) - len(shape) - 1)
        self._new_count += x.shape[0]
        self._new_sum += torch.sum(x, dim=0)
        self._new_sum_sq += torch.sum(torch.square(x), dim=0)

    def update(self) -> None:
        """
        Update the normalizer statistics based on recorded data.
        """
        if self._mean_sq is None:
            self._mean_sq = self._calc_mean_sq(self._mean, self._std)
        new_count = self._new_count
        new_mean = self._new_sum / new_count
        new_mean_sq = self._new_sum_sq / new_count
        new_total = self._count + new_count
        w_old = self._count.type(torch.float) / new_total.type(torch.float)
        w_new = float(new_count) / new_total.type(torch.float)
        self._mean[:] = w_old * self._mean + w_new * new_mean
        self._mean_sq[:] = w_old * self._mean_sq + w_new * new_mean_sq
        self._count[:] = new_total
        self._std[:] = self._calc_std(self._mean, self._mean_sq)
        self._new_count = 0
        self._new_sum[:] = 0
        self._new_sum_sq[:] = 0

    def get_shape(self) -> Tuple[int]:
        """
        Returns the shape of the normalizer.

        Returns:
            Tuple[int]: Shape of the normalizer.
        """
        return self._mean.shape

    def get_count(self) -> int:
        """
        Returns the count of recorded samples.

        Returns:
            int: Count of recorded samples.
        """
        return self._count.item()

    def get_mean(self) -> torch.Tensor:
        """
        Returns the mean tensor of the normalizer.

        Returns:
            torch.Tensor: Mean tensor.
        """
        return self._mean

    def get_std(self) -> torch.Tensor:
        """
        Returns the standard deviation tensor of the normalizer.

        Returns:
            torch.Tensor: Standard deviation tensor.
        """
        return self._std

    def set_mean_std(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """
        Sets the mean and standard deviation tensors of the normalizer.

        Args:
            mean (torch.Tensor): Mean tensor.
            std (torch.Tensor): Standard deviation tensor.
        """
        assert mean.shape == self._mean.shape and std.shape == self._std.shape, f"Normalizer shape mismatch. Expecting {self._mean.shape}, but got {mean.shape} and {std.shape}."
        self._mean[:] = mean
        self._std[:] = std
        self._mean_sq[:] = self._calc_mean_sq(self._mean, self._std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the given tensor.

        Args:
            x (torch.Tensor): Tensor to normalize.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        norm_x = (x - self._mean) / (self._std + self._eps)
        norm_x = torch.clamp(norm_x, -self._clip, self._clip)
        return norm_x.type(self.dtype)

    def unnormalize(self, norm_x: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize the given tensor.

        Args:
            norm_x (torch.Tensor): Tensor to unnormalize.

        Returns:
            torch.Tensor: Unnormalized tensor.
        """
        x = norm_x * self._std + self._mean
        return x.type(self.dtype)

    def _calc_std(self, mean: torch.Tensor, mean_sq: torch.Tensor) -> torch.Tensor:
        var = mean_sq - torch.square(mean)
        var = torch.clamp_min(var, 0.0)
        return torch.sqrt(var).type(self.dtype)

    def _calc_mean_sq(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.square(std) + torch.square(mean)
        return mean_sq.type(self.dtype)

    def _build_params(self, shape: Tuple[int], device: torch.device, init_mean: Optional[torch.Tensor], init_std: Optional[torch.Tensor]) -> None:
        self._count = torch.nn.Parameter(torch.zeros([1], device=device, requires_grad=False, dtype=torch.long), requires_grad=False)
        self._mean = torch.nn.Parameter(torch.zeros(shape, device=device, requires_grad=False, dtype=self.dtype), requires_grad=False)
        self._std = torch.nn.Parameter(torch.ones(shape, device=device, requires_grad=False, dtype=self.dtype), requires_grad=False)
        if init_mean is not None:
            assert init_mean.shape == shape, f"Normalizer init mean shape mismatch. Expecting {shape}, but got {init_mean.shape}."
            self._mean[:] = init_mean
        if init_std is not None:
            assert init_std.shape == shape, f"Normalizer init std shape mismatch. Expecting {shape}, but got {init_std.shape}."
            self._std[:] = init_std
        self._mean_sq = None
        self._new_count = 0
        self._new_sum = torch.zeros_like(self._mean)
        self._new_sum_sq = torch.zeros_like(self._mean)
