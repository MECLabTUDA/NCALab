#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca import (
    GrowingNCADataset,
    GrowingNCAModel,
    train_basic_nca,
    WEIGHTS_PATH,
    get_compute_device,
)

import click

import torch
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import numpy as np

from pilmoji import Pilmoji
from PIL import Image, ImageFont


def get_emoji_image(emoji: str = "🦎", padding: int = 2, size: int = 24):
    """_summary_

    Args:
        emoji (str, optional): String containing a single emoji character. Defaults to "🦎".
        padding (int, optional): Number of pixels to pad to the sides. Defaults to 2.
        size (int, optional): Total image size without padding. Defaults to 24.

    Returns:
        Image: Output PIL.Image containing an emoji on transparent background.
    """
    dims = (padding * 2 + size, padding * 2 + size)
    with Image.new("RGBA", dims, (255, 255, 255, 0)) as image:
        font = ImageFont.truetype("arial.ttf", size)
        with Pilmoji(image) as pilmoji:
            pilmoji.text((padding, padding), emoji.strip(), (0, 0, 0), font)
        return image


def train_growing_emoji(batch_size: int, hidden_channels: int):
    """_summary_

    Args:
        batch_size (int, optional): _description_.
        hidden_channels (int, optional): _description_.
    """
    writer = SummaryWriter()

    device = get_compute_device("cuda:0")

    nca = GrowingNCAModel(
        device,
        num_image_channels=4,
        num_hidden_channels=hidden_channels,
        use_alive_mask=False,
    )

    image = np.asarray(get_emoji_image())
    dataset = GrowingNCADataset(image, nca.num_channels, batch_size=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    train_basic_nca(
        nca, WEIGHTS_PATH / "growing_emoji.pth", loader, summary_writer=writer
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=12, type=int)
def main(batch_size, hidden_channels):
    train_growing_emoji(batch_size=batch_size, hidden_channels=hidden_channels)


if __name__ == "__main__":
    main()
