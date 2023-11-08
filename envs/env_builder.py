from typing import Any, Dict, Union

import torch
import yaml

import envs.atari_env as atari_env
import envs.box2d_env as box2d_env
import envs.classic_env as classic_env
import envs.env_dm as env_dm
import envs.mujoco_env as mujoco_env


def build_env(env_file: str, device: Union[str, torch.device], visualize: bool) -> atari_env.AtariEnv:
    """
    Builds and returns an environment of the specified type based on the configuration provided in the env_file.

    Args:
        env_file (str): The path to the environment configuration file.
        device (Union[str, torch.device]): The device (CPU/GPU) on which to perform computations.
        visualize (bool): Whether to visualize the environment or not.

    Returns:
        atari_env.AtariEnv: The constructed Atari environment.

    Raises:
        AssertionError: If the specified environment is not supported.
    """

    env_config = load_env_file(env_file)
    env_name = env_config["env_name"]
    print(f"Building {env_name} env")

    if env_name.startswith("atari_"):
        env = atari_env.AtariEnv(config=env_config, device=device, visualize=visualize)
    elif env_name.startswith("classic_"):
        env = classic_env.ClassicEnv(config=env_config, device=device, visualize=visualize)
    elif env_name.startswith("dm_"):
        env = env_dm.DMEnv(config=env_config, device=device, visualize=visualize)
    elif env_name.startswith("mujoco_"):
        env = mujoco_env.MujocoEnv(config=env_config, device=device, visualize=visualize)
    elif env_name.startswith("box2d_"):
        env = box2d_env.Box2DEnv(config=env_config, device=device, visualize=visualize)
    else:
        raise ValueError(f"Unsupported env: {env_name}")

    return env


def load_env_file(file: str) -> Dict[str, Any]:
    """
    Loads the environment configuration file and returns it as a dictionary.

    Args:
        file (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """

    with open(file, "r") as stream:
        env_config = yaml.safe_load(stream)

    return env_config
