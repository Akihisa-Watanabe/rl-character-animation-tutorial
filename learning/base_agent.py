import abc
import enum
import os
import time
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import torch

import envs.base_env as base_env
import learning.distribution_gaussian_diag as distribution_gaussian_diag
import learning.experience_buffer as experience_buffer
import learning.normalizer as normalizer
import learning.return_tracker as return_tracker
import util.tb_logger as tb_logger
import util.torch_util as torch_util


class AgentMode(enum.Enum):
    """
    Enumeration for agent modes.
    """

    TRAIN = 0
    TEST = 1


class BaseAgent(torch.nn.Module):
    """
    Abstract base class for an agent.

    Attributes:
        NAME (str): Name of the agent.
    """

    NAME = "base"

    def __init__(self, config: Dict, env: base_env.BaseEnv, device: torch.device) -> None:
        """
        Initializes the agent.

        Args:
            config (Dict): Configuration dictionary.
            env (base_env.BaseEnv): Environment object.
            device (torch.device): Device object.
        """
        super().__init__()

        self._env: base_env.BaseEnv = env
        self._device: torch.device = device
        self._iter: int = 0
        self._sample_count: int = 0
        self._load_params(config)

        self._build_normalizers()
        self._build_model(config)
        self._build_optimizer(config)

        self._build_exp_buffer(config)
        self._build_return_tracker()

        self._mode: AgentMode = AgentMode.TRAIN
        self._curr_obs: Optional[torch.Tensor] = None
        self._curr_info: Optional[Dict] = None

        self.to(self._device)

    def train_model(self, max_samples: int, out_model_file: str, int_output_dir: str, log_file: str) -> None:
        """
        Train the agent's model.

        Args:
            max_samples (int): Maximum number of samples to train on.
            out_model_file (str): File to save the trained model.
            int_output_dir (str): Directory for intermittent model outputs.
            log_file (str): File for logging.

        Returns:
            None
        """
        start_time = time.time()

        self._curr_obs, self._curr_info = self._env.reset()
        self._logger = self._build_logger(log_file)
        self._init_train()

        test_info = None
        while self._sample_count < max_samples:
            train_info = self._train_iter()

            output_iter = self._iter % self._iters_per_output == 0
            if output_iter:
                test_info = self.test_model(self._test_episodes)

            self._sample_count = self._update_sample_count()
            self._log_train_info(train_info, test_info, start_time)
            self._logger.print_log()

            if output_iter:
                self._logger.write_log()
                self._output_train_model(self._iter, out_model_file, int_output_dir)

                self._train_return_tracker.reset()
                self._curr_obs, self._curr_info = self._env.reset()

            self._iter += 1

    def test_model(self, num_episodes: int) -> Dict:
        """
        Test the agent's model.

        Args:
            num_episodes (int): Number of episodes for testing.

        Returns:
            Dict: Test information including mean return, mean episode length, and episodes.
        """
        self.eval()
        self.set_mode(AgentMode.TEST)

        self._curr_obs, self._curr_info = self._env.reset()
        test_info = self._rollout_test(num_episodes)

        return test_info

    def get_action_size(self) -> int:
        """
        Get the size of the action space.

        Returns:
            int: Size of the action space.
        """
        a_space = self._env.get_action_space()
        if isinstance(a_space, gym.spaces.Box):
            a_size = np.prod(a_space.shape)
        elif isinstance(a_space, gym.spaces.Discrete):
            a_size = 1
        else:
            assert False, "Unsuppoted action space: {}".format(a_space)
        return a_size

    def set_mode(self, mode: AgentMode) -> None:
        """
        Sets the agent mode (Train/Test).

        Args:
            mode (AgentMode): The agent mode.

        Returns:
            None
        """
        self._mode = mode
        if self._mode == AgentMode.TRAIN:
            self._env.set_mode(base_env.EnvMode.TRAIN)
        elif self._mode == AgentMode.TEST:
            self._env.set_mode(base_env.EnvMode.TEST)
        else:
            assert False, "Unsupported agent mode: {}".format(mode)

    def save(self, out_file: str) -> None:
        """
        Saves the agent model to a file.

        Args:
            out_file (str): File path to save the model.

        Returns:
            None
        """
        state_dict = self.state_dict()
        torch.save(state_dict, out_file)

    def load(self, in_file: str) -> None:
        """
        Loads the agent model from a file.

        Args:
            in_file (str): File path to load the model from.

        Returns:
            None
        """
        state_dict = torch.load(in_file, map_location=self._device)
        self.load_state_dict(state_dict)

    def _load_params(self, config: Dict) -> None:
        """
        Load parameters from the configuration dictionary.

        Args:
            config (Dict): Configuration dictionary.

        Returns:
            None
        """
        self._discount = config["discount"]
        self._steps_per_iter = config["steps_per_iter"]
        self._iters_per_output = config["iters_per_output"]
        self._test_episodes = config["test_episodes"]
        self._normalizer_samples = config.get("normalizer_samples", np.inf)

    @abc.abstractmethod
    def _build_model(self, config: Dict) -> None:
        """
        Abstract method to build the model.

        Args:
            config (Dict): Configuration dictionary.

        Returns:
            None
        """
        pass

    def _build_normalizers(self) -> None:
        """
        Build normalizers for observation and action spaces.

        Returns:
            None
        """
        obs_space = self._env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_norm = normalizer.Normalizer(obs_space.shape, device=self._device, dtype=obs_dtype)
        self._a_norm = self._build_action_normalizer()

    def _build_action_normalizer(self) -> normalizer.Normalizer:
        """
        Build and return a normalizer for the action space of the environment.

        Returns:
            normalizer.Normalizer: The action space normalizer.

        Raises:
            AssertionError: If the action space type is not supported.
        """
        a_space = self._env.get_action_space()
        a_dtype = torch_util.numpy_dtype_to_torch(a_space.dtype)

        if isinstance(a_space, gym.spaces.Box):
            a_mean = torch.tensor(0.5 * (a_space.high + a_space.low), device=self._device, dtype=a_dtype)
            a_std = torch.tensor(0.5 * (a_space.high - a_space.low), device=self._device, dtype=a_dtype)
            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, init_std=a_std, dtype=a_dtype)
        elif isinstance(a_space, gym.spaces.Discrete):
            a_mean = torch.tensor([0], device=self._device, dtype=a_dtype)
            a_std = torch.tensor([1], device=self._device, dtype=a_dtype)
            a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, init_std=a_std, eps=0, dtype=a_dtype)
        else:
            assert False, "Unsuppoted action space: {}".format(a_space)
        return a_norm

    def _build_optimizer(self, config: Dict) -> None:
        """
        Build optimizer for the agent.

        Args:
            config (Dict): Configuration dictionary.

        Returns:
            None
        """
        lr = float(config["learning_rate"])
        params = list(self.parameters())
        params = [p for p in params if p.requires_grad]
        self._optimizer = torch.optim.SGD(params, lr, momentum=0.9)

    def _build_exp_buffer(self, config: Dict) -> None:
        """
        Build experience buffer for the agent.

        Args:
            config (Dict): Configuration dictionary.

        Returns:
            None
        """
        buffer_length = self._get_exp_buffer_length()
        self._exp_buffer = experience_buffer.ExperienceBuffer(buffer_length=buffer_length, device=self._device)

        obs_space = self._env.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        obs_buffer = torch.zeros([buffer_length] + list(obs_space.shape), device=self._device, dtype=obs_dtype)
        self._exp_buffer.add_buffer("obs", obs_buffer)

        next_obs_buffer = torch.zeros_like(obs_buffer)
        self._exp_buffer.add_buffer("next_obs", next_obs_buffer)

        a_space = self._env.get_action_space()
        a_dtype = torch_util.numpy_dtype_to_torch(a_space.dtype)
        action_buffer = torch.zeros([buffer_length] + list(a_space.shape), device=self._device, dtype=a_dtype)
        self._exp_buffer.add_buffer("action", action_buffer)

        reward_buffer = torch.zeros([buffer_length], device=self._device, dtype=torch.float)
        self._exp_buffer.add_buffer("reward", reward_buffer)

        done_buffer = torch.zeros([buffer_length], device=self._device, dtype=torch.int)
        self._exp_buffer.add_buffer("done", done_buffer)

    def _build_return_tracker(self) -> None:
        """
        Build return tracker for the agent.

        Returns:
            None
        """
        self._train_return_tracker = return_tracker.ReturnTracker(self._device)

    @abc.abstractmethod
    def _get_exp_buffer_length(self) -> int:
        """
        Abstract method to get the length of the experience buffer.

        Returns:
            int: Length of the experience buffer.
        """
        return 0

    def _build_logger(self, log_file: str) -> tb_logger.TBLogger:
        """
        Build logger for the agent.

        Args:
            log_file (str): File for logging.

        Returns:
            tb_logger.TBLogger: Logger object.
        """
        log = tb_logger.TBLogger()
        log.set_step_key("Samples")
        log.configure_output_file(log_file)
        return log

    def _update_sample_count(self) -> int:
        """
        Update the sample count for the agent.

        Returns:
            int: Updated sample count.
        """
        return self._exp_buffer.get_total_samples()

    def _init_train(self) -> None:
        """
        Initialize training for the agent.

        Returns:
            None
        """
        self._iter = 0
        self._sample_count = 0
        self._exp_buffer.clear()
        self._train_return_tracker.reset()

    def _train_iter(self) -> Dict:
        """
        Execute a single training iteration for the agent.

        Returns:
            Dict: Training information including mean return, mean episode length, and episodes.
        """
        self._init_iter()

        self.eval()
        self.set_mode(AgentMode.TRAIN)
        self._rollout_train(self._steps_per_iter)

        data_info = self._build_train_data()
        train_info = self._update_model()

        if self._need_normalizer_update():
            self._update_normalizers()

        info = {**train_info, **data_info}

        info["mean_return"] = self._train_return_tracker.get_mean_return().item()
        info["mean_ep_len"] = self._train_return_tracker.get_mean_ep_len().item()
        info["episodes"] = self._train_return_tracker.get_episodes()

        return info

    def _init_iter(self) -> None:
        """
        Initialize variables or states needed at the beginning of each training iteration.

        Returns:
            None
        """
        return

    def _rollout_train(self, num_steps: int) -> None:
        """
        Execute a training rollout for a given number of steps. This involves making decisions,
        taking actions, updating trackers, and recording data.

        Args:
            num_steps (int): Number of steps for which to run the training rollout.

        Returns:
            None
        """
        for i in range(num_steps):
            action, action_info = self._decide_action(self._curr_obs, self._curr_info)
            self._record_data_pre_step(self._curr_obs, self._curr_info, action, action_info)

            next_obs, r, done, next_info = self._step_env(action)
            self._train_return_tracker.update(r, done)
            self._record_data_post_step(next_obs, r, done, next_info)

            self._curr_obs = next_obs

            if done != base_env.DoneFlags.NULL.value:
                self._curr_obs, self._curr_info = self._env.reset()
            self._exp_buffer.inc()

    def _rollout_test(self, num_episodes: int) -> Dict[str, Any]:
        """
        Roll out the agent in test mode for a given number of episodes and collect statistics.

        Args:
            num_episodes (int): The number of episodes to roll out.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - 'mean_return': The mean return over the test episodes.
                - 'mean_ep_len': The mean episode length over the test episodes.
                - 'episodes': The number of test episodes.

        Notes:
            This method updates the `_curr_obs` and `_curr_info` class attributes to hold the most
            recent observation and info returned by the environment.
        """
        sum_rets = 0.0
        sum_ep_lens = 0

        self._curr_obs, self._curr_info = self._env.reset()

        for e in range(num_episodes):
            curr_ret = 0.0
            curr_ep_len = 0
            while True:
                action, action_info = self._decide_action(self._curr_obs, self._curr_info)
                self._curr_obs, r, done, self._curr_info = self._step_env(action)

                curr_ret += r.item()
                curr_ep_len += 1

                if done != base_env.DoneFlags.NULL.value:
                    sum_rets += curr_ret
                    sum_ep_lens += curr_ep_len
                    self._curr_obs, self._curr_info = self._env.reset()
                    break

        mean_return = sum_rets / max(1, num_episodes)
        mean_ep_len = sum_ep_lens / max(1, num_episodes)

        test_info = {"mean_return": mean_return, "mean_ep_len": mean_ep_len, "episodes": num_episodes}
        return test_info

    @abc.abstractmethod
    def _decide_action(self, obs: torch.Tensor, info: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Abstract method to decide the action given an observation and additional info.

        Args:
            obs (torch.Tensor): Current observation.
            info (Dict): Additional information.

        Returns:
            Tuple[torch.Tensor, Dict]: Chosen action and additional info about the action.
        """
        a = None
        a_info = dict()
        return a, a_info

    def _step_env(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, Dict]:
        """
        Step the environment forward using the given action.

        Args:
            action (torch.Tensor): Action to take in the environment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int, Dict]:
                Next observation, reward, done flag, and additional information.
        """
        obs, r, done, info = self._env.step(action)
        return obs, r, done, info

    def _record_data_pre_step(self, obs: torch.Tensor, info: Dict, action: torch.Tensor, action_info: Dict) -> None:
        """
        Record data before taking a step in the environment.

        Args:
            obs (torch.Tensor): Current observation.
            info (Dict): Additional information.
            action (torch.Tensor): Chosen action.
            action_info (Dict): Additional information about the action.

        Returns:
            None
        """
        self._exp_buffer.record("obs", obs)
        self._exp_buffer.record("action", action)

        if self._need_normalizer_update():
            self._obs_norm.record(obs.unsqueeze(0))

    def _record_data_post_step(self, next_obs: torch.Tensor, r: torch.Tensor, done: int, next_info: Dict) -> None:
        """
        Record data after taking a step in the environment.

        Args:
            next_obs (torch.Tensor): Next observation.
            r (torch.Tensor): Reward obtained.
            done (int): Done flag.
            next_info (Dict): Additional information after the step.

        Returns:
            None
        """
        self._exp_buffer.record("next_obs", next_obs)
        self._exp_buffer.record("reward", r)
        self._exp_buffer.record("done", done)

    def _reset_done_envs(self, done: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Reset the environment for episodes that are done.

        Args:
            done (torch.Tensor): Flags indicating whether each episode is done.

        Returns:
            Tuple[torch.Tensor, Dict]: New observations and additional information after reset.
        """
        done_indices = (done != base_env.DoneFlags.NULL.value).nonzero(as_tuple=False)
        done_indices = torch.flatten(done_indices)
        obs, info = self._env.reset(done_indices)
        return obs, info

    def _need_normalizer_update(self) -> bool:
        """
        Determine whether the normalizers need to be updated.

        Returns:
            bool: True if the normalizers need to be updated, False otherwise.
        """
        return self._sample_count < self._normalizer_samples

    def _update_normalizers(self) -> None:
        """
        Update the normalizers based on the data collected.

        Returns:
            None
        """
        self._obs_norm.update()

    def _build_train_data(self) -> Dict:
        """
        Build the training data based on the current experience buffer.

        Returns:
            Dict: Dictionary containing the training data.
        """
        return dict()

    @abc.abstractmethod
    def _update_model(self) -> Dict:
        """
        Abstract method to update the model based on the training data.

        Returns:
            Dict: Dictionary containing the training metrics.
        """
        return

    def _compute_succ_val(self) -> float:
        """
        Compute the "success" value, which is the expected value of successful outcomes.

        Returns:
            float: The success value.
        """
        r_succ = self._env.get_reward_succ()
        val_succ = r_succ / (1.0 - self._discount)
        return val_succ

    def _compute_fail_val(self) -> float:
        """
        Compute the "failure" value, which is the expected value of failed outcomes.

        Returns:
            float: The failure value.
        """
        r_fail = self._env.get_reward_fail()
        val_fail = r_fail / (1.0 - self._discount)
        return val_fail

    def _compute_val_bound(self) -> Tuple[float, float]:
        """
        Compute the value bounds, which are the minimum and maximum expected values.

        Returns:
            Tuple[float, float]: A tuple containing the minimum and maximum expected values.
        """
        r_min, r_max = self._env.get_reward_bounds()
        val_min = r_min / (1.0 - self._discount)
        val_max = r_max / (1.0 - self._discount)
        return val_min, val_max

    def _log_train_info(self, train_info: Dict, test_info: Dict, start_time: float) -> None:
        """
        Log training and testing information for the current iteration.

        Args:
            train_info (Dict): Dictionary containing training metrics.
            test_info (Dict): Dictionary containing testing metrics.
            start_time (float): The time when training started.

        Returns:
            None
        """
        wall_time = (time.time() - start_time) / (60 * 60)  # store time in hours
        self._logger.log("Iteration", self._iter, collection="1_Info")
        self._logger.log("Wall_Time", wall_time, collection="1_Info")
        self._logger.log("Samples", self._sample_count, collection="1_Info")

        test_return = test_info["mean_return"]
        test_ep_len = test_info["mean_ep_len"]
        test_eps = test_info["episodes"]
        self._logger.log("Test_Return", test_return, collection="0_Main")
        self._logger.log("Test_Episode_Length", test_ep_len, collection="0_Main", quiet=True)
        self._logger.log("Test_Episodes", test_eps, collection="1_Info", quiet=True)

        train_return = train_info.pop("mean_return")
        train_ep_len = train_info.pop("mean_ep_len")
        train_eps = train_info.pop("episodes")
        self._logger.log("Train_Return", train_return, collection="0_Main")
        self._logger.log("Train_Episode_Length", train_ep_len, collection="0_Main", quiet=True)
        self._logger.log("Train_Episodes", train_eps, collection="1_Info", quiet=True)

        for k, v in train_info.items():
            val_name = k.title()
            if torch.is_tensor(v):
                v = v.item()
            self._logger.log(val_name, v)

    def _compute_action_bound_loss(self, norm_a_dist: Any) -> Optional[torch.Tensor]:
        """
        Compute the loss due to action bounds violations, if applicable.

        Args:
            norm_a_dist (Any): Normalized action distribution.

        Returns:
            Optional[torch.Tensor]: Loss due to action bounds violations, or None if not applicable.
        """
        loss = None
        action_space = self._env.get_action_space()
        if isinstance(action_space, gym.spaces.Box):
            a_low = action_space.low
            a_high = action_space.high
            valid_bounds = np.all(np.isfinite(a_low)) and np.all(np.isfinite(a_high))

            if valid_bounds:
                assert isinstance(norm_a_dist, distribution_gaussian_diag.DistributionGaussianDiag)
                # assume that actions have been normalized between [-1, 1]
                bound_min = -1
                bound_max = 1
                violation_min = torch.clamp_max(norm_a_dist.mode - bound_min, 0.0)
                violation_max = torch.clamp_min(norm_a_dist.mode - bound_max, 0)
                violation = torch.sum(torch.square(violation_min), dim=-1) + torch.sum(torch.square(violation_max), dim=-1)
                loss = violation

        return loss

    def _output_train_model(self, iter: int, out_model_file: str, int_output_dir: str) -> None:
        """
        Output the trained model at the current iteration.

        Args:
            iter (int): Current iteration number.
            out_model_file (str): File to save the trained model.
            int_output_dir (str): Directory for intermittent model outputs.

        Returns:
            None
        """
        self.save(out_model_file)

        if int_output_dir != "":
            int_model_file = os.path.join(int_output_dir, "model_{:010d}.pt".format(iter))
            self.save(int_model_file)
