#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca import (
    ClassificationNCAModel,
    train_basic_nca,
    WEIGHTS_PATH,
    show_batch_classification,
    get_compute_device,
)

import click

import torch

from medmnist import PathMNIST

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def train_selfclass_pathmnist(
    batch_size: int,
    hidden_channels: int,
    gpu,
    gpu_index,
):
    writer = SummaryWriter()

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    dataset_train = PathMNIST(
        split="train",
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # automatic divide by 255
                transforms.RandomErasing(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        ),
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True, batch_size=batch_size
    )

    dataset_val = PathMNIST(
        split="val",
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # automatic divide by 255
                transforms.RandomErasing(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        ),
    )
    loader_val = torch.utils.data.DataLoader(dataset_val, shuffle=True, batch_size=256)

    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        hidden_size=128,
        num_classes=9,
        use_alive_mask=False,
        fire_rate=0.5,
    )
    train_basic_nca(
        nca,
        WEIGHTS_PATH / "selfclass_pathmnist.pth",
        loader_train,
        loader_val,
        summary_writer=writer,
        plot_function=show_batch_classification,
        batch_repeat=2,
        max_iterations=100000,
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=16, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def main(batch_size, hidden_channels, gpu: bool, gpu_index: int):
    train_selfclass_pathmnist(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
