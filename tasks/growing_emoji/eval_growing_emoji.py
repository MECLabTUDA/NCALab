#!/usr/bin/env python3
import os
import sys
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import (
    GrowingNCAModel,
    get_compute_device,
    print_NCALab_banner,
    fix_random_seed,
    Animator,
)

import click

import torch


TASK_PATH = Path(__file__).parent
FIGURE_PATH = TASK_PATH / "figures"
FIGURE_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH = TASK_PATH / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)


@click.command()
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def eval_growing_emoji(gpu: bool, gpu_index: int):
    print_NCALab_banner()
    fix_random_seed()

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    nca = GrowingNCAModel(
        device,
        num_image_channels=4,
        num_hidden_channels=12,
        use_alive_mask=True,
    ).to(device)

    nca.load_state_dict(
        torch.load(
            WEIGHTS_PATH / "ncalab_growing_emoji" / "last_model.pth",
            weights_only=True,
        )
    )

    seed = nca.make_seed(48, 48)
    animator = Animator(nca, seed)

    out_path = FIGURE_PATH / "growing_emoji.gif"
    animator.save(out_path)
    click.secho(f"Done. You'll find the generated GIF in {out_path} .")


if __name__ == "__main__":
    eval_growing_emoji()
