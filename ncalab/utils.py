from __future__ import annotations
import os
import random
from typing import Any

import numpy as np

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import BasicNCAModel


def get_compute_device(device: str = "cuda:0") -> torch.device:
    """
    Obtain a pytorch compute device handle based on input string.
    If user tries to get a CUDA device, but none is available,
    defaults to CPU.

    :param device: Device string, defaults to "cuda:0".
    :type device: str

    :returns: Pytorch compute device.
    :rtype: torch.device
    """
    if device == "cpu":
        return torch.device("cpu")
    d = torch.device(device if torch.cuda.is_available() else "cpu")
    return d


def pad_input(
    x: torch.Tensor,
    nca: "BasicNCAModel",
    noise: bool = True,
    mean: float = 0.5,
    std: float = 0.225,
) -> torch.Tensor:
    """
    Pads the BCWH input tensor along its channel dimension to match the expected number of
    channels required by the NCA model. Pads with either Gaussian noise (parameterized by
    mean and std) or zeros, depending on the "noise" parameter.

    :param x: Input image tensor, BCWH.
    :type x: torch.Tensor
    :param nca: NCA model definition.
    :type nca: ncalab.BasicNCAModel
    :param noise: Whether to pad with noise. Otherwise zeros, defaults to True.
    :type noise: bool, optional
    :param mean: Mean (mu) of Gaussian noise distribution, defaults to 0.5.
    :type mean: float, optional
    :param std: Standard deviation (sigma) of Gaussian noise distribution, defaults to 0.225.
    :type std: float, optional

    :returns: Input tensor, BCWH, padded along the channel dimension.
    :rtype: torch.Tensor
    """
    if x.shape[1] < nca.num_channels:
        x = F.pad(
            x, (0, 0, 0, 0, 0, nca.num_channels - x.shape[1], 0, 0), mode="constant"
        )
        if noise:
            x[
                :,
                nca.num_image_channels : nca.num_image_channels
                + nca.num_hidden_channels,
                :,
                :,
            ] = torch.normal(
                mean,
                std,
                size=(x.shape[0], nca.num_hidden_channels, x.shape[2], x.shape[3]),
            )
    return x


def print_NCALab_banner():
    """
    Show NCALab banner on terminal.
    """
    banner = """
 _   _  _____          _           _
| \\ | |/ ____|   /\\   | |         | |
|  \\| | |       /  \\  | |     __ _| |__
| . ` | |      / /\\ \\ | |    / _` | '_ \\
| |\\  | |____ / ____ \\| |___| (_| | |_) |
|_| \\_|\\_____/_/    \\_\\______\\__,_|_.__/
-----------------------------------------
    Developed at MECLab - TU Darmstadt
-----------------------------------------
    """
    print(banner)


def print_mascot(message: str):
    """
    Show help text in a speech bubble.

    :param message: Message to display.
    :type message: str
    """
    if not message:
        return
    w = max([len(L) for L in message.splitlines()])
    print("  " + "-" * w)
    for L in message.splitlines():
        print(f"| {L}" + " " * (w - len(L)) + " |")
    print("  " + "=" * w)
    print(" " * w + "   \\")
    print(" " * w + "    \\")

    try:
        print(" " * (w + 3) + "\N{MICROSCOPE}\N{RAT}")
    except UnicodeEncodeError:
        print(" " * (w + 5) + ":3")


"""
Default random seed to use within this project.
"""
DEFAULT_RANDOM_SEED = 1337


def fix_random_seed(seed: int = DEFAULT_RANDOM_SEED):
    """
    Fixes the random seed for all pseudo-random number generators,
    including Python-native, Numpy and Pytorch.

    :param seed: Random seed, defaults to DEFAULT_RANDOM_SEED.
    :type seed: int, optional
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap(x: Any):
    """
    Panics if x is None, otherwise returns x.

    This is a useful shorthand for cases such as ``x = unwrap(some_object).do_something()``
    in which we are 99% certain that some_object is not None and want to avoid a mypy complaint.

    :param x: Any kind of object.
    :type x: Any
    :raises RuntimeError: If x is None.
    :return: Just passes through the input x if it is not None.
    """
    if x is None:
        raise RuntimeError("unwrap() failed: Expected return other than None.")
    return x
