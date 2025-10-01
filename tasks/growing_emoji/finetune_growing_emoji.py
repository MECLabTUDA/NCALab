#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import sys
from pathlib import Path

import click
import numpy as np
import torch
from growing_utils import get_emoji_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import (  # noqa: E402
    BasicNCATrainer,
    GrowingNCADataset,
    GrowingNCAModel,
    fix_random_seed,
    get_compute_device,
    print_mascot,
    print_NCALab_banner,
)

TASK_PATH = Path(__file__).parent.resolve()
WEIGHTS_PATH = TASK_PATH / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)


def finetune_growing_emoji(
    batch_size: int, hidden_channels: int, gpu: bool, gpu_index: int
):
    print_NCALab_banner()
    print_mascot(
        "You're about to finetune a pre-trained NCA\n"
        "by adjusting only its final layer.\n"
    )
    print()

    fix_random_seed()
    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    nca = GrowingNCAModel(
        device,
        num_image_channels=4,
        num_hidden_channels=hidden_channels,
        use_alive_mask=True,
        num_learned_filters=0,
        fire_rate=0.5,
    )

    # Create emoji dataset for initial training
    image_lizard = np.asarray(get_emoji_image("\N{LIZARD}"))
    dataset_lizard = GrowingNCADataset(
        image_lizard, nca.num_channels, batch_size=batch_size
    )
    loader_lizard = DataLoader(
        dataset_lizard, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # Run initial training
    writer = SummaryWriter(comment="Growing Emoji: Pre-Training")
    trainer = BasicNCATrainer(nca, WEIGHTS_PATH / "growing_emoji_finetuned")
    trainer.max_epochs = 5000
    trainer.train(loader_lizard, summary_writer=writer, save_every=500)
    writer.close()

    # So far so good.
    # Now we will freeze the first layer, and only train the linear layer! :)

    # Idea: shuffle first layer's weights. They'll still follow the same distribution.
    with torch.no_grad():
        W = nca.network[0].weight.data.clone()
        W = W.view(-1)
        np.random.shuffle(W.cpu().numpy())
        nca.network[0].weight.data.copy_(W.view(nca.network[0].weight.data.size()))

    # Create emoji dataset for finetuning
    image_dna = np.asarray(get_emoji_image("\N{RAT}"))
    dataset_dna = GrowingNCADataset(image_dna, nca.num_channels, batch_size=batch_size)
    loader_dna = DataLoader(dataset_dna, batch_size=batch_size, shuffle=False)

    # Re-train with frozen final layer
    nca.finetune()
    writer = SummaryWriter(comment="Growing Emoji: Finetuning")
    trainer.max_epochs = 5000  # we don't need as many iterations now
    trainer.train(loader_dna, summary_writer=writer, save_every=100)
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
    finetune_growing_emoji(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
