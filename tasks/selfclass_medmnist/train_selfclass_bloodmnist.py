import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca.models.classificationNCA import ClassificationNCAModel
from nca.training import train_basic_nca
from nca.paths import WEIGHTS_PATH
from nca.visualization import show_batch_classification

import click

import torch

from medmnist import BloodMNIST

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def train_selfclass_bloodmnist(batch_size: int, hidden_channels: int):
    writer = SummaryWriter()

    device = torch.device("cuda:0")

    dataset_train = BloodMNIST(
        split="train",
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=batch_size)

    dataset_val = BloodMNIST(
        split="val",
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    loader_val = torch.utils.data.DataLoader(dataset_val, shuffle=True, batch_size=batch_size)

    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=dataset_train.labels.shape[1],
    )
    train_basic_nca(
        nca,
        WEIGHTS_PATH / "selfclass_bloodmnist.pth",
        loader_train,
        loader_val,
        summary_writer=writer,
        plot_function=show_batch_classification,
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=9, type=int)
def main(batch_size, hidden_channels):
    train_selfclass_bloodmnist(batch_size=batch_size, hidden_channels=hidden_channels)


if __name__ == "__main__":
    main()
