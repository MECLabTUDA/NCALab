import os
import random

import numpy as np

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]


def get_compute_device(device: str = "cuda:0") -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    d = torch.device(device if torch.cuda.is_available() else "cpu")
    return d


def pad_input(x, nca, noise=True):
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


def print_mascot(message):
    if not message:
        return
    h = len(message.splitlines())
    w = max([len(L) for L in message.splitlines()])
    print("  " + "-" * w)
    for L in message.splitlines():
        print(f"| {L}" + " " * (w - len(L)) + " |")
    print("  " + "=" * w)
    print(" " * w + "   \\")
    print(" " * w + "    \\")

    try:
        print(" " * (w + 3) + "\N{Microscope}\N{rat}")
    except UnicodeEncodeError:
        print(" " * (w + 5) + ":3")


DEFAULT_RANDOM_SEED = 1337


def fix_random_seed(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_Kfold_idx(data, K):
    N_fold = len(data) // K
    idx = np.arange(len(data))
    folds = []
    for i in range(K):
        val_idx = idx[i * N_fold : (i + 1) * N_fold]
        train_idx = np.concatenate(
            [idx[: i * N_fold], idx[(i + 1) * N_fold :]]
        )
        folds.append((train_idx, val_idx))
    return folds
