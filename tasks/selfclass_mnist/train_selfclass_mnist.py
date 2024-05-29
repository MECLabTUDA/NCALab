import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(root_dir)
sys.path.append(root_dir)

from nca.models.classificationNCA import ClassificationNCAModel
from nca.training import train_basic_nca
from nca.paths import WEIGHTS_PATH

import click

import torch

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


def train_selfclass_mnist(batch_size=8, hidden_chanels=9):
    writer = SummaryWriter()

    mnist = MNIST(
        "mnist",
        download=True,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    loader = torch.utils.data.DataLoader(mnist, shuffle=True, batch_size=batch_size)
    device = torch.device("cuda:0")

    nca = ClassificationNCAModel(
        device,
        num_image_channels=1,
        num_hidden_channels=hidden_chanels,
        num_classes=10,
        fire_rate=0.5,
        hidden_size=128,
        use_alive_mask=False,
        immutable_image_channels=True,
        learned_filters=2,
    )
    train_basic_nca(nca, loader, WEIGHTS_PATH / "selfclass_mnist.pth", summary_writer=writer)
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=9, type=int)
def main(batch_size, hidden_channels):
    train_selfclass_mnist(batch_size=batch_size, hidden_chanels=hidden_channels)


if __name__ == "__main__":
    main()
