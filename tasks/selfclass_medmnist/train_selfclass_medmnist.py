import click

from .train_selfclass_bloodmnist import train_selfclass_bloodmnist
from .train_selfclass_pathmnist import train_selfclass_pathmnist


def train_selfclass_medmnist(batch_size: int, hidden_channels: int):
    train_selfclass_bloodmnist(batch_size, hidden_channels)
    train_selfclass_pathmnist(batch_size, hidden_channels)


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=9, type=int)
def main(batch_size, hidden_channels):
    train_selfclass_medmnist(batch_size=batch_size, hidden_channels=hidden_channels)


if __name__ == "__main__":
    main()
