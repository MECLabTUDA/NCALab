import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca.models.classificationNCA import ClassificationNCAModel
from nca.training import train_basic_nca
from nca.paths import WEIGHTS_PATH
from nca.visualization import show_batch_binary_image_classification

import click

import torch

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def train_selfclass_mnist(batch_size=8, hidden_channels=9):
    writer = SummaryWriter()

    mnist_train = MNIST(
        "mnist",
        download=True,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    train_indices, val_indices, _, _ = train_test_split(
        range(len(mnist_train)),
        mnist_train.targets,
        stratify=mnist_train.targets,
        test_size=int(len(mnist_train) * 0.2),
    )

    train_split = Subset(mnist_train, train_indices)
    val_split = Subset(mnist_train, val_indices)

    loader_train = torch.utils.data.DataLoader(
        train_split, shuffle=True, batch_size=batch_size
    )
    loader_val = torch.utils.data.DataLoader(
        val_split, shuffle=True, batch_size=batch_size
    )
    device = torch.device("cuda:0")

    nca = ClassificationNCAModel(
        device,
        num_image_channels=1,
        num_hidden_channels=hidden_channels,
        num_classes=10,
    )

    train_basic_nca(
        nca,
        WEIGHTS_PATH / "selfclass_mnist.pth",
        loader_train,
        loader_val,
        summary_writer=writer,
        plot_function=show_batch_binary_image_classification,
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=9, type=int)
def main(batch_size, hidden_channels):
    train_selfclass_mnist(batch_size=batch_size, hidden_channels=hidden_channels)


if __name__ == "__main__":
    main()
