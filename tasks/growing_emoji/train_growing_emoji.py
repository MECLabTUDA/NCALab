import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca.data import GrowingNCADataset
from nca.models.growingNCA import GrowingNCAModel
from nca.training import train_basic_nca
from nca.paths import WEIGHTS_PATH

import click

import torch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import numpy as np

from pilmoji import Pilmoji
from PIL import Image, ImageFont


def get_emoji_image(emoji="🦎", padding=2, size=24):
    dims = (padding * 2 + size, padding * 2 + size)
    with Image.new("RGBA", dims, (255, 255, 255, 0)) as image:
        font = ImageFont.truetype("arial.ttf", size)
        with Pilmoji(image) as pilmoji:
            pilmoji.text((padding, padding), emoji.strip(), (0, 0, 0), font)
        return image


def train_growing_emoji(batch_size=8, hidden_channels=12):
    writer = SummaryWriter()

    device = torch.device("cuda:0")

    nca = GrowingNCAModel(
        device,
        num_image_channels=4,
        num_hidden_channels=hidden_channels,
        fire_rate=0.5,
        hidden_size=128,
        use_alive_mask=False,
        learned_filters=0,
    )

    image = np.asarray(get_emoji_image())
    dataset = GrowingNCADataset(image, nca.num_channels, batch_size=8)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    train_basic_nca(
        nca, loader, WEIGHTS_PATH / "growing_emoji.pth", summary_writer=writer
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=16, type=int)
def main(batch_size, hidden_channels):
    train_growing_emoji(batch_size=batch_size, hidden_channels=hidden_channels)


if __name__ == "__main__":
    main()
