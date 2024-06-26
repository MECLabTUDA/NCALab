import torch
import numpy as np


def get_compute_device(device: str = "cuda:0") -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    d = torch.device(device if torch.cuda.is_available() else "cpu")
    return d


def pad_input(x, nca, noise=True):
    if x.shape[1] < nca.num_channels:
        x = np.pad(
            x,
            [
                (0, 0),  # batch
                (0, nca.num_channels - x.shape[1]),  # channels
                (0, 0),  # width
                (0, 0),  # height
            ],
            mode="constant",
        )
        if noise:
            x[
                :,
                nca.num_image_channels : nca.num_image_channels + nca.num_hidden_channels,
                :,
                :,
            ] = np.random.normal(
                size=(x.shape[0], nca.num_hidden_channels, x.shape[2], x.shape[3])
            )
    return x
