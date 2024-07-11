#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca import (
    GrowingNCADataset,
    GrowingNCAModel,
    BasicNCATrainer,
    WEIGHTS_PATH,
    get_compute_device,
)

import click

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


def train_growing_emoji(
    batch_size: int, hidden_channels: int, gpu: bool, gpu_index: int
):
    """_summary_

    Args:
        batch_size (int, optional): _description_.
        hidden_channels (int, optional): _description_.
    """
    writer = SummaryWriter()

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    nca = GrowingNCAModel(
        device,
        num_image_channels=4,
        num_hidden_channels=hidden_channels,
        use_alive_mask=False,
    )

    image_lizard = np.asarray(get_emoji_image("🦎"))
    dataset_lizard = GrowingNCADataset(image_lizard, nca.num_channels, batch_size=batch_size)
    loader_lizard = DataLoader(dataset_lizard, batch_size=batch_size, shuffle=False)

    image_dna = np.asarray(get_emoji_image("🧬"))
    dataset_dna = GrowingNCADataset(image_dna, nca.num_channels, batch_size=batch_size)
    loader_dna = DataLoader(dataset_dna, batch_size=batch_size, shuffle=False)

    trainer = BasicNCATrainer(nca, WEIGHTS_PATH / "growing_emoji_finetuned.pth")
    trainer.train_basic_nca(
        loader_lizard, summary_writer=writer, save_every=100
    )
    nca.finetune()

    writer = SummaryWriter()
    trainer.train_basic_nca(
        loader_dna, summary_writer=writer, save_every=100
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=12, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def main(batch_size, hidden_channels, gpu, gpu_index):
    train_growing_emoji(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
