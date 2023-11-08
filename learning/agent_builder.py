from typing import Any, Dict, Union

import hw2.baseline_pg_agent as baseline_pg_agent
import hw2.gae_pg_agent as gae_pg_agent
import hw2.rtg_pg_agent as rtg_pg_agent
import hw2.vanilla_pg_agent as vanilla_pg_agent
import yaml


def build_agent(agent_file: str, env: Any, device: Any) -> Any:
    """
    Constructs an agent based on the configuration provided in the agent_file.

    Args:
        agent_file (str): Path to the agent configuration file.
        env (Any): The environment the agent will interact with.
        device (Any): Device on which the agent's computations will be performed.

    Returns:
        Any: An instance of the desired agent.
    """
    agent_config = load_agent_file(agent_file)

    agent_name = agent_config["agent_name"]
    print(f"Building {agent_name} agent")

    if agent_name == baseline_pg_agent.BaselinePGAgent.NAME:
        agent = baseline_pg_agent.BaselinePGAgent(config=agent_config, env=env, device=device)
    elif agent_name == rtg_pg_agent.RtgPGAgent.NAME:
        agent = rtg_pg_agent.RtgPGAgent(config=agent_config, env=env, device=device)
    elif agent_name == vanilla_pg_agent.VanillaPGAgent.NAME:
        agent = vanilla_pg_agent.VanillaPGAgent(config=agent_config, env=env, device=device)
    elif agent_name == gae_pg_agent.GAEPGAgent.NAME:
        agent = gae_pg_agent.GAEPGAgent(config=agent_config, env=env, device=device)
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    return agent


def load_agent_file(file: str) -> Dict[str, Union[str, Dict]]:
    """
    Loads the agent configuration from a YAML file.

    Args:
        file (str): Path to the agent configuration file.

    Returns:
        Dict[str, Union[str, Dict]]: The parsed agent configuration.
    """
    with open(file, "r") as stream:
        agent_config = yaml.safe_load(stream)
    return agent_config
