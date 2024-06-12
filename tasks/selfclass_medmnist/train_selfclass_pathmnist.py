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


def train_selfclass_pathmnist(batch_size: int, hidden_channels: int):
    writer = SummaryWriter()

    device = get_compute_device("cuda:0")

    dataset_train = PathMNIST(
        split="train",
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True, batch_size=batch_size
    )

    dataset_val = PathMNIST(
        split="val",
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    loader_val = torch.utils.data.DataLoader(dataset_val, shuffle=True, batch_size=128)

    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        hidden_size=128,
        num_classes=9,
    )
    train_basic_nca(
        nca,
        WEIGHTS_PATH / "selfclass_pathmnist.pth",
        loader_train,
        loader_val,
        summary_writer=writer,
        plot_function=show_batch_classification,
        batch_repeat=2,
        gradient_clipping=False,
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=20, type=int)
def main(batch_size, hidden_channels):
    train_selfclass_pathmnist(batch_size=batch_size, hidden_channels=hidden_channels)


if __name__ == "__main__":
    main()
