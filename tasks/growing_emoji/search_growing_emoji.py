#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path

import click
from torch.utils.data import DataLoader
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import (
    GrowingNCADataset,
    GrowingNCAModel,
    get_compute_device,
    NCALab_banner,
    print_mascot,
    ParameterSearch,
    ParameterSet,
)

from growing_utils import get_emoji_image

TASK_PATH = Path(__file__).parent


def search_growing_emoji(
    batch_size: int, hidden_channels: int, gpu: bool, gpu_index: int
):
    logging.basicConfig(level=logging.INFO)

    """
    Main function to run the "growing emoji search" example task.
    """
    # Display prologue
    NCALab_banner()
    print_mascot(
        "You are about to run a hyperparameter search.\n"
        "\n"
        "Grab a cup of coffee, this will take some time."
    )
    print()

    # Select device, try to use GPU or fall back to CPU
    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    # Create dataset containing a single growing emoji
    image = np.asarray(get_emoji_image())
    dataset = GrowingNCADataset(image, hidden_channels + 4, batch_size=batch_size)
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Set up parameter ranges for grid search
    model_params = ParameterSet(
        # grid search parameters (iterable)
        fire_rate=[0.2, 0.5, 0.8],
        num_learned_filters=[0, 2],
        dx_noise=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
        # fixed parameters
        num_image_channels=4,
        num_hidden_channels=hidden_channels,
        use_alive_mask=True,
    )
    trainer_params = ParameterSet(
        # grid search parameters (iterable)
        lr=[1e-3, 16e-4],
        # fixed parameters
        max_epochs=100
    )

    # Set up hyperparameter search (grid search)
    search = ParameterSearch(device, GrowingNCAModel, model_params, trainer_params)

    # Print search metadata
    print(search.info())
    print()

    # Run the search!
    df = search(dataloader_train)
    print(df)
    df.to_csv(TASK_PATH / "search_summary_growing_emoji.csv")


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
    search_growing_emoji(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
