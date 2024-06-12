#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca import GrowingNCAModel, WEIGHTS_PATH, get_compute_device

import click

import torch

import matplotlib.pyplot as plt


@click.command()
def eval_growing_emoji():
    device = get_compute_device("cuda:0")

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
