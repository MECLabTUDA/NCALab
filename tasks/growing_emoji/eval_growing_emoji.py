#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca import GrowingNCAModel, WEIGHTS_PATH, get_compute_device

import click

import torch

import matplotlib.pyplot as plt


@click.command()
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def eval_growing_emoji(gpu: bool, gpu_index: int):
    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    nca = GrowingNCAModel(
        device,
        num_image_channels=4,
        num_hidden_channels=12,
        use_alive_mask=False,
    ).to(device)
    nca.load_state_dict(torch.load(WEIGHTS_PATH / "growing_emoji.pth"))
    nca.eval()

    image = nca.grow(32, 32)

    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    eval_growing_emoji()
