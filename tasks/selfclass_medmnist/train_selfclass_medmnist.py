import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca.models.classificationNCA import ClassificationNCAModel
from nca.training import train_basic_nca
from nca.paths import WEIGHTS_PATH

import click

import torch

from medmnist import PathMNIST

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def train_selfclass_medmnist(batch_size=8, hidden_channels=9):
    writer = SummaryWriter()

    device = torch.device("cuda:0")

    dataset = PathMNIST(
        split="train",
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    num_classes = 9

    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=num_classes,
    )
    train_basic_nca(
        nca,
        WEIGHTS_PATH / "selfclass_medmnist.pth",
        loader,
        summary_writer=writer,
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=9, type=int)
def main(batch_size, hidden_channels):
    train_selfclass_medmnist(batch_size=batch_size, hidden_channels=hidden_channels)


if __name__ == "__main__":
    main()
