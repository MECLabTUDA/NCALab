#!/usr/bin/env python3
import click

from .train_selfclass_bloodmnist import train_selfclass_bloodmnist
from .train_selfclass_pathmnist import train_selfclass_pathmnist


def train_selfclass_medmnist(
    batch_size: int,
    hidden_channels: int,
    gpu: bool,
    gpu_index: int,
):
    train_selfclass_bloodmnist(batch_size, hidden_channels, gpu, gpu_index)
    train_selfclass_pathmnist(batch_size, hidden_channels, gpu, gpu_index)


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=9, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def main(batch_size, hidden_channels, gpu, gpu_index):
    train_selfclass_medmnist(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
