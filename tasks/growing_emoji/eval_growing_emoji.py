#!/usr/bin/env python3
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import GrowingNCAModel, WEIGHTS_PATH, get_compute_device

import click

import torch

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.animation as animation  # type: ignore[import-untyped]


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
        use_alive_mask=True,
    ).to(device)

    nca.load_state_dict(
        torch.load(WEIGHTS_PATH / "ncalab_growing_emoji.pth", weights_only=True)
    )
    nca.eval()

    images = nca.grow(48, 48, steps=100, save_steps=True)

    fig, ax = plt.subplots()
    fig.set_size_inches(2, 2)
    im = ax.imshow(images[0], animated=True)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.tight_layout()

    def update(i):
        im.set_array(images[i])
        return (im,)

    animation_fig = animation.FuncAnimation(
        fig,
        update,
        frames=len(images),
        interval=10,
        blit=True,
        repeat=True,
        repeat_delay=3000,
    )
    animation_fig.save("artwork/growing_emoji.gif")


if __name__ == "__main__":
    eval_growing_emoji()
