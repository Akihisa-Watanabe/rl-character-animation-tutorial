from typing import Any, Dict, Union

import numpy as np
import torch
from torch import nn


def add_torch_dict(in_dict: Dict[str, torch.Tensor], out_dict: Dict[str, torch.Tensor]) -> None:
    """
    Add tensors from one dictionary to another.

    Args:
        in_dict (Dict[str, torch.Tensor]): Dictionary containing tensors to add.
        out_dict (Dict[str, torch.Tensor]): Dictionary where tensors will be added to.
    """
    for k, v in in_dict.items():
        if v.requires_grad:
            v = v.detach()

        if k in out_dict:
            out_dict[k] += v
        else:
            out_dict[k] = v


def scale_torch_dict(scale: float, out_dict: Dict[str, torch.Tensor]) -> None:
    """
    Scale all tensors in a dictionary by a given scale factor.

    Args:
        scale (float): Scale factor.
        out_dict (Dict[str, torch.Tensor]): Dictionary containing tensors to be scaled.
    """
    for k in out_dict.keys():
        out_dict[k] *= scale


def calc_layers_out_size(layers: nn.Module) -> int:
    """
    Calculate the output size of the last layer in a neural network module.

    Args:
        layers (nn.Module): Neural network module.

    Returns:
        int: Output size of the last layer.
    """
    modules = list(layers.modules())
    for m in reversed(modules):
        if hasattr(m, "out_features"):
            out_size = m.out_features
            break
    return out_size


def torch_dtype_to_numpy(torch_dtype: torch.dtype) -> np.dtype:
    """
    Convert a PyTorch dtype to a NumPy dtype.

    Args:
        torch_dtype (torch.dtype): PyTorch data type.

    Returns:
        np.dtype: Corresponding NumPy data type.
    """
    if torch_dtype == torch.float32:
        numpy_dtype = np.float32
    elif torch_dtype == torch.float64:
        numpy_dtype = np.float64
    elif torch_dtype == torch.uint8:
        numpy_dtype = np.uint8
    elif torch_dtype == torch.int64:
        numpy_dtype = np.int64
    else:
        raise ValueError(f"Unsupported type {torch_dtype}")
    return numpy_dtype


def numpy_dtype_to_torch(numpy_dtype: np.dtype) -> torch.dtype:
    """
    Convert a NumPy dtype to a PyTorch dtype.

    Args:
        numpy_dtype (np.dtype): NumPy data type.

    Returns:
        torch.dtype: Corresponding PyTorch data type.
    """
    if numpy_dtype == np.float32:
        torch_dtype = torch.float32
    elif numpy_dtype == np.float64:
        torch_dtype = torch.float64
    elif numpy_dtype == np.uint8:
        torch_dtype = torch.uint8
    elif numpy_dtype == np.int64:
        torch_dtype = torch.int64
    else:
        raise ValueError(f"Unsupported type {numpy_dtype}")
    return torch_dtype


class UInt8ToFloat(nn.Module):
    """
    A PyTorch Module that converts an input tensor of dtype uint8 to float32 and scales it by 1/255.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the conversion and scaling.

        Args:
            x (torch.Tensor): Input tensor of dtype uint8.

        Returns:
            torch.Tensor: Output tensor of dtype float32.
        """
        float_x = x.type(torch.float32) / 255.0
        return float_x
