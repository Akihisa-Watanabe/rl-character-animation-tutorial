import collections
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch


class ExperienceBuffer:
    """
    Example Usage:
        # Initialize an ExperienceBuffer object to hold 100 experiences, using CPU for storage
        >>> exp_buffer = ExperienceBuffer(100, torch.device('cpu'))

        # Define and add a buffer to hold 4-dimensional state vectors. The buffer can accommodate 100 such states.
        >>> exp_buffer.add_buffer('states', torch.zeros((100, 4)))

        # Define and add a buffer to hold 1-dimensional action values. This buffer can also hold 100 actions.
        >>> exp_buffer.add_buffer('actions', torch.zeros((100, 1)))

        # Record multiple experiences into the buffer.
        >>> for i in range(100):
        ...     new_state = torch.tensor([1.0, 0.0, -1.0, 0.5])  # Simulated state
        ...     new_action = torch.tensor([0])  # Simulated action
        ...
        ...     # Record the state and action into the buffer
        ...     exp_buffer.record('states', new_state)
        ...     exp_buffer.record('actions', new_action)
        ...
        ...     # Move the buffer head to the next position
        ...     exp_buffer.inc()

        # Randomly sample a batch of experiences for training. Here, we sample three experiences.
        >>> sampled_experience = exp_buffer.sample(3)

        # Retrieve the total number of experiences recorded so far.
        >>> total_experiences = exp_buffer.get_total_samples()

    """

    def __init__(self, buffer_length: int, device: torch.device):
        """
        Initializes the ExperienceBuffer class with a given buffer length and device.

        Args:
            buffer_length (int): The total length (number of slots) the buffer can hold.
            device (torch.device): The computing device [CPU/GPU] where the buffer will be stored.
        """

        # Internal variables to hold buffer properties
        self._buffer_length = buffer_length
        self._device = device

        self._buffer_head = 0
        self._total_samples = 0

        self._buffers = dict()
        self._sample_buf = torch.randperm(self._buffer_length, device=self._device, dtype=torch.long)

        # Initialize the sampling buffer
        self._sample_buf_head = 0
        self._reset_sample_buf()  # Fill it with random permutation at the start

    def add_buffer(self, name, buffer):
        """
        Adds a new buffer tensor to the existing set of buffers.

        Args:
            name (str): Name/identifier for the buffer.
            buffer (torch.Tensor): The tensor that will be used as a buffer.

        Raises:
            AssertionError: If the buffer doesn't meet the specified requirements.
        """
        assert len(buffer.shape) >= 1, "The buffer should have at least 1 dimension."
        assert buffer.shape[0] == self._buffer_length, "The first dimension of the buffer should match the buffer_length."
        assert name not in self._buffers, f"A buffer with the name '{name}' already exists."

        self._buffers[name] = buffer

    def reset(self):
        """
        Resets the head of the buffer to the start and also resets the sampling buffer.
        """
        self._buffer_head = 0
        self._reset_sample_buf()

    def clear(self):
        """
        Clears all the data in the buffer and resets the total sample count to zero.
        """
        self.reset()
        self._total_samples = 0

    def inc(self):
        """
        Increments the buffer head and increases the total sample count.
        """
        self._buffer_head = (self._buffer_head + 1) % self._buffer_length
        self._total_samples += 1

    def get_total_samples(self) -> int:
        """
        Retrieves the total number of samples that have been recorded.

        Returns:
            int: Total number of samples recorded.
        """

        return self._total_samples

    def get_sample_count(self) -> int:
        """
        Calculates the effective number of samples that can be sampled.

        Returns:
            int: Number of samples that can be effectively sampled.
        """
        sample_count = min(self._total_samples, self._buffer_length)
        return sample_count

    def record(self, name: str, data: Union[int, float, torch.Tensor]):
        """
        Records a new piece of data into a named buffer.

        Args:
            name (str): The name of the buffer where the data will be recorded.
            data (Union[int, float, torch.Tensor]): The new data sample.
        """
        data_buf = self._buffers[name]
        data_buf[self._buffer_head] = data

    def set_data(self, name: str, data: torch.Tensor):
        """
        Sets an entire buffer with new data.

        Args:
            name (str): The name of the buffer that will be updated.
            data (torch.Tensor): The new data that will replace the existing buffer.

        Raises:
            AssertionError: If the data shape doesn't match the existing buffer's shape.
        """
        data_buf = self._buffers[name]
        assert data_buf.shape[0] == data.shape[0], "The first dimension of the data should match the buffer."

        data_buf[:] = data

    def get_data(self, name: str) -> torch.Tensor:
        """
        Retrieves the data stored in a named buffer.

        Args:
            name (str): The name of the buffer.

        Returns:
            torch.Tensor: The data stored in the named buffer.
        """
        return self._buffers[name]

    def sample(self, n: int) -> Dict[str, torch.Tensor]:
        """
        Samples 'n' experiences from the buffer.

        Args:
            n (int): The number of samples to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the sampled data.
        """
        output = dict()
        rand_idx = self._sample_rand_idx(n)

        for key, data in self._buffers.items():
            batch_data = data[rand_idx]
            output[key] = batch_data

        return output

    def _reset_sample_buf(self):
        """
        Internal method to reset the sample buffer used for random sampling.
        """
        self._sample_buf[:] = torch.randperm(self._buffer_length, device=self._device, dtype=torch.long)
        self._sample_buf_head = 0

    def _sample_rand_idx(self, n):
        """
        Internal method to obtain 'n' random indices for sampling from the buffer.

        Args:
            n (int): The number of random indices needed.

        Returns:
            torch.Tensor: A tensor containing 'n' random indices.

        Raises:
            AssertionError: If 'n' is more than the buffer length.
        """
        buffer_len = self._sample_buf.shape[0]
        assert n <= buffer_len, "Requested more samples than available in buffer."

        # Logic to get 'n' random indices from the sample buffer
        if self._sample_buf_head + n <= buffer_len:
            rand_idx = self._sample_buf[self._sample_buf_head : self._sample_buf_head + n]
            self._sample_buf_head += n
        else:
            rand_idx0 = self._sample_buf[self._sample_buf_head :]
            remainder = n - (buffer_len - self._sample_buf_head)

            self._reset_sample_buf()
            rand_idx1 = self._sample_buf[:remainder]
            rand_idx = torch.cat([rand_idx0, rand_idx1], dim=0)

            self._sample_buf_head = remainder

        sample_count = self.get_sample_count()
        rand_idx = torch.remainder(rand_idx, sample_count)
        return rand_idx
