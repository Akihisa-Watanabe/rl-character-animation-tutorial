import torch

import envs.base_env as base_env


class ReturnTracker:
    def __init__(self, device: torch.device):
        """
        Tracker for returns (rewards) from an environment.

        Args:
            device (torch.device): Device to allocate tensors.
        """
        num_envs = 1
        self._device = device
        self._episodes = 0
        self._mean_return = torch.zeros([1], device=device, dtype=torch.float32)
        self._mean_ep_len = torch.zeros([1], device=device, dtype=torch.float32)

        self._return_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._ep_len_buf = torch.zeros([num_envs], device=device, dtype=torch.long)
        self._eps_per_env_buf = torch.zeros([num_envs], device=device, dtype=torch.long)

    def get_mean_return(self) -> torch.Tensor:
        """Returns the mean return."""
        return self._mean_return

    def get_mean_ep_len(self) -> torch.Tensor:
        """Returns the mean episode length."""
        return self._mean_ep_len

    def get_episodes(self) -> int:
        """Returns the number of episodes."""
        return self._episodes

    def get_eps_per_env(self) -> torch.Tensor:
        """Returns the number of episodes per environment."""
        return self._eps_per_env_buf

    def reset(self) -> None:
        """Resets the tracker's statistics."""
        self._episodes = 0
        self._eps_per_env_buf[:] = 0
        self._mean_return[:] = 0.0
        self._mean_ep_len[:] = 0.0
        self._return_buf[:] = 0.0
        self._ep_len_buf[:] = 0

    def update(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        """
        Updates the tracker with new reward and done signals.

        Args:
            reward (torch.Tensor): Rewards from the environment.
            done (torch.Tensor): Done signals from the environment.
        """
        assert reward.shape == self._return_buf.shape
        assert done.shape == self._return_buf.shape

        self._return_buf += reward
        self._ep_len_buf += 1

        reset_mask = done != base_env.DoneFlags.NULL.value
        reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
        num_resets = len(reset_ids)

        if num_resets > 0:
            new_mean_return = torch.mean(self._return_buf[reset_ids])
            new_mean_ep_len = torch.mean(self._ep_len_buf[reset_ids].type(torch.float))

            new_count = self._episodes + num_resets
            w_new = float(num_resets) / new_count
            w_old = float(self._episodes) / new_count

            self._mean_return = w_new * new_mean_return + w_old * self._mean_return
            self._mean_ep_len = w_new * new_mean_ep_len + w_old * self._mean_ep_len
            self._episodes += num_resets

            self._return_buf[reset_ids] = 0.0
            self._ep_len_buf[reset_ids] = 0
            self._eps_per_env_buf[reset_ids] += 1
