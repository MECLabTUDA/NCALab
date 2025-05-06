import os
import random

import numpy as np

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]


def get_compute_device(device: str = "cuda:0") -> torch.device:
    """
    Obtain a pytorch compute device handle based on input string.
    If user tries to get a CUDA device, but none is available,
    defaults to CPU.

    :param device [str]: Device string. Defaults to "cuda:0".

    :returns [torch.device]: Pytorch compute device.
    """
    if device == "cpu":
        return torch.device("cpu")
    d = torch.device(device if torch.cuda.is_available() else "cpu")
    return d


def pad_input(x, nca, noise: bool = True):
    """
    Pads input tensor along channel dimension to match the expected number of
    channels required by the NCA model. Pads with either Gaussian noise or zeros,
    depending on "noise" parameter. Gaussian noise has mean of 0.5 and sigma 0.225.

    :param x [torch.Tensor]: Input image tensor, BCWH.
    :param nca [BasicNCAModel]: NCA model definition.
    :param noise [bool]: Whether to pad with noise. Otherwise zeros. Defaults to True.

    :returns: Input tensor, BCWH, padded along the channel dimension.
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
                0.5,
                0.225,
                size=(x.shape[0], nca.num_hidden_channels, x.shape[2], x.shape[3]),
            )
    return x


def NCALab_banner():
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

    :param message [str]: Message to display.
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
__DEFAULT_RANDOM_SEED = 1337


def fix_random_seed(seed: int = __DEFAULT_RANDOM_SEED):
    """
    Fixes the random seed for all pseudo-random number generators,
    including Python-native, Numpy and Pytorch.

    :param seed [int]: . Defaults to DEFAULT_RANDOM_SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
