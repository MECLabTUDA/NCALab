import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import (
    ClassificationNCAModel,
    BasicNCATrainer,
    WEIGHTS_PATH,
    show_batch_binary_image_classification,
    get_compute_device,
)

import click

from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

import torch

from torchvision.datasets import MNIST  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]


def eval_selfclass_mnist(
    batch_size: int, hidden_channels: int, gpu: bool, gpu_index: int
):
    mnist_test = MNIST(
        "mnist",
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    loader_test = torch.utils.data.DataLoader(
        mnist_test, shuffle=True, batch_size=batch_size
    )

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    nca = ClassificationNCAModel(
        device,
        num_image_channels=1,
        num_hidden_channels=hidden_channels,
        num_classes=10,
        pixel_wise_loss=True,
    )


@click.command()
@click.option("--hidden-channels", "-H", default=9, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def main(batch_size, hidden_channels, gpu, gpu_index):
    eval_selfclass_mnist(
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
