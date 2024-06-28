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
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter


def train_selfclass_pathmnist(
    batch_size: int,
    hidden_channels: int,
    gpu: bool,
    gpu_index: int,
    lambda_activity: float,
):
    writer = SummaryWriter(
        comment=f"Lambda.activity_{lambda_activity}_channels.hidden_{hidden_channels}"
    )

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    T = transforms.Compose(
        [
            v2.ToImageTensor(),
            v2.ConvertImageDtype(dtype=torch.float32),
            # transforms.Normalize(
            #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    dataset_train = PathMNIST(split="train", download=True, transform=T)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True, batch_size=batch_size
    )

    dataset_val = PathMNIST(split="val", download=True, transform=T)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=32,
    )

    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        hidden_size=128,
        num_classes=9,
        use_alive_mask=False,
        fire_rate=0.5,
        lambda_activity=lambda_activity,
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
        gradient_clipping=False,
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
@click.option("--lambda-activity", type=float, default=0.0)
def main(
    batch_size, hidden_channels, gpu: bool, gpu_index: int, lambda_activity: float
):
    train_selfclass_pathmnist(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
        lambda_activity=lambda_activity,
    )


if __name__ == "__main__":
    main()
