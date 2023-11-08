from typing import Dict, Tuple, Union

import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.pg_model as pg_model
import util.torch_util as torch_util


class RtgPGAgent(base_agent.BaseAgent):
    NAME = "RtgPG"

    def __init__(self, config: Dict[str, Union[int, float, str]], env: base_env.BaseEnv, device: torch.device) -> None:
        """
        Initializes the RtgPGAgent object.

        Args:
            config (Dict): A dictionary containing various configuration parameters.
            env (base_env.BaseEnv): The environment in which the agent will operate.
            device (torch.device): The device (cpu or cuda) on which the computations will be performed.
        """
        super().__init__(config, env, device)

    def _load_params(self, config: Dict) -> None:
        """
        Load parameters from the config.

        Args:
            config (Dict): A dictionary containing various configuration parameters.
        """
        super()._load_params(config)

        self._batch_size = config["batch_size"]
        self._norm_Q_clip = config["norm_Q_clip"]
        self._action_bound_weight = config["action_bound_weight"]
        self._norm = config["norm"]

    def _build_model(self, config: Dict) -> None:
        """
        Builds the model using the provided configuration.

        Args:
            config (Dict): A dictionary containing various configuration parameters.
        """
        model_config = config["model"]
        self._model = pg_model.PGModel(model_config, self._env)

    def _get_exp_buffer_length(self) -> int:
        """
        Retrieves the length of the experience buffer.

        Returns:
            int: The length of the experience buffer.
        """
        return self._steps_per_iter

    def _build_exp_buffer(self, config: Dict) -> None:
        """
        Builds the experience buffer using the provided configuration.

        Args:
            config (Dict): A dictionary containing various configuration parameters.
        """
        super()._build_exp_buffer(config)

        buffer_length = self._get_exp_buffer_length()
        tar_val_buffer = torch.zeros([buffer_length], device=self._device, dtype=torch.float)
        self._exp_buffer.add_buffer("tar_val", tar_val_buffer)

        Q_buffer = torch.zeros([buffer_length], device=self._device, dtype=torch.float)
        self._exp_buffer.add_buffer("Q", Q_buffer)

        norm_obs_buffer = torch.zeros_like(self._exp_buffer.get_data("obs"))
        self._exp_buffer.add_buffer("norm_obs", norm_obs_buffer)

        norm_action_buffer = torch.zeros_like(self._exp_buffer.get_data("action"))
        self._exp_buffer.add_buffer("norm_action", norm_action_buffer)

    def _init_iter(self) -> None:
        """
        Initializes the iterator.
        """
        super()._init_iter()
        self._exp_buffer.reset()

    def _build_optimizer(self, config: Dict) -> None:
        """
        Builds the optimizer using the provided configuration.

        Args:
            config (Dict): A dictionary containing various configuration parameters.
        """
        actor_lr = float(config["actor_learning_rate"])
        actor_params = list(self._model._actor_layers.parameters()) + list(self._model._action_dist.parameters())
        actor_params_grad = [p for p in actor_params if p.requires_grad]
        self._actor_optimizer = torch.optim.Adam(actor_params_grad, actor_lr)

    def _decide_action(self, obs: torch.Tensor, info: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Decides the action to be taken based on the observations and info provided.

        Args:
            obs (torch.Tensor): The observations.
            info (Dict): Additional information.

        Returns:
            Tuple[torch.Tensor, Dict]: The decided action and additional info.
        """
        norm_obs = self._obs_norm.normalize(obs)
        norm_action_dist = self._model.eval_actor(norm_obs)

        if self._mode == base_agent.AgentMode.TRAIN:
            norm_a = norm_action_dist.sample()
        elif self._mode == base_agent.AgentMode.TEST:
            norm_a = norm_action_dist.mode
        else:
            assert False, "Unsupported agent mode: {}".format(self._mode)

        norm_a = norm_a.detach()
        a = self._a_norm.unnormalize(norm_a)

        info = dict()

        return a, info

    def _build_train_data(self) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Builds the training data from the experience buffer.

        Returns:
            Dict: A dictionary containing information about the training data such as Q_mean and Q_std.
        """
        self.eval()

        obs = self._exp_buffer.get_data("obs")
        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        action = self._exp_buffer.get_data("action")
        norm_action = self._a_norm.normalize(action)
        norm_obs = self._obs_norm.normalize(obs)
        ret = self._calc_return(r, done)
        Q = ret

        Q_std, Q_mean = torch.std_mean(Q)
        norm_Q = (Q - Q_mean) / torch.clamp_min(Q_std, 1e-5)
        norm_Q = torch.clamp(norm_Q, -self._norm_Q_clip, self._norm_Q_clip)

        self._exp_buffer.set_data("tar_val", ret)
        if self._norm:
            self._exp_buffer.set_data("Q", norm_Q)
        else:
            self._exp_buffer.set_data("Q", Q)
        self._exp_buffer.set_data("norm_obs", norm_obs)
        self._exp_buffer.set_data("norm_action", norm_action)

        info = {"Q_mean": Q_mean, "Q_std": Q_std}
        return info

    def _update_model(self) -> Dict[str, torch.Tensor]:
        """
        Updates the model using the training data.

        Returns:
            Dict: A dictionary containing information about the training process.
        """
        self.train()

        num_samples = self._exp_buffer.get_sample_count()
        batch_size = self._batch_size
        num_batches = int(np.ceil(float(num_samples) / batch_size))
        train_info = dict()

        torch_util.scale_torch_dict(1.0 / num_batches, train_info)

        actor_batch = {"norm_obs": self._exp_buffer.get_data("norm_obs"), "norm_action": self._exp_buffer.get_data("norm_action"), "Q": self._exp_buffer.get_data("Q")}
        actor_info = self._update_actor(actor_batch)

        for key, data in actor_info.items():
            train_info[key] = data

        return train_info

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Updates the actor using the provided batch of data.

        Args:
            batch (Dict): A dictionary containing normalized observations, normalized actions, and Q-values.

        Returns:
            Dict: A dictionary containing information about the actor update process, such as actor_loss.
        """
        norm_obs = batch["norm_obs"]
        norm_a = batch["norm_action"]
        Q = batch["Q"]

        loss = self._calc_actor_loss(norm_obs, norm_a, Q)

        info = {"actor_loss": loss}

        if self._action_bound_weight != 0:
            a_dist = self._model.eval_actor(norm_obs)
            action_bound_loss = self._compute_action_bound_loss(a_dist)
            if action_bound_loss is not None:
                action_bound_loss = torch.mean(action_bound_loss)
                loss += self._action_bound_weight * action_bound_loss
                info["action_bound_loss"] = action_bound_loss.detach()

        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()

        return info

    def _calc_return(self, r: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """
        Calculates the return values based on rewards and done flags.

        Args:
            r (torch.Tensor): A tensor containing the rewards.
            done (torch.Tensor): A tensor containing the done flags, indicating whether an episode has ended.

        Returns:
            torch.Tensor: A tensor containing the calculated return values.
        """
        return_t = torch.zeros_like(r)
        return_t[done] = r[done]
        episode_ends = torch.where(done)[0]
        self.n_episodes = len(episode_ends)
        start_idx = 0
        for end_idx in episode_ends:
            # Calculate reward-to-go within the episode
            for t in reversed(range(start_idx, end_idx + 1)):
                if t == end_idx:
                    continue  # Skip, as we have already set this above
                return_t[t] = r[t] + self._discount * return_t[t + 1]
            start_idx = end_idx + 1  # Update the starting index for the next episode

        return return_t

    def _calc_actor_loss(self, norm_obs: torch.Tensor, norm_a: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Calculates the actor loss.

        Args:
            norm_obs (torch.Tensor): Normalized observations.
            norm_a (torch.Tensor): Normalized actions.
            Q (torch.Tensor): Q-values.

        Returns:
            torch.Tensor: The calculated actor loss.
        """
        a_dist = self._model.eval_actor(norm_obs)
        log_probs = a_dist.log_prob(norm_a)
        loss = (-log_probs * Q).sum() / self.n_episodes
        return loss
