from typing import Dict, Tuple, Type, Union

import numpy as np
import torch
from torch.nn import Module

from learning.nets import *


def build_net(net_name: str, input_dict: Dict[str, np.ndarray], activation: Union[Module, Type[Module]] = torch.nn.ReLU()) -> Tuple[Module, Dict]:
    """
    Builds a neural network using the specified network function name.

    Args:
        net_name (str): The name of the network function to use.
        input_dict (Dict[str, np.ndarray]): Dictionary containing the input tensors.
        activation (Union[torch.nn.Module, Type[torch.nn.Module]]): The activation function to use in the network. Defaults to ReLU.

    Returns:
        Tuple[torch.nn.Module, Dict]: The constructed neural network and additional information as a dictionary.

    Raises:
        AssertionError: If the specified network function name is not supported.
    """
    if net_name in globals():
        net_func = globals()[net_name]
        net, info = net_func.build_net(input_dict, activation)
    else:
        assert False, f"Unsupported net: {net_name}"
    return net, info
