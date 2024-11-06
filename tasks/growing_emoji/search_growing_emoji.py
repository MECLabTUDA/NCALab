#!/usr/bin/env python3
import sys, os
import logging

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

import click
from torch.utils.data import DataLoader
import numpy as np

from growing_utils import get_emoji_image


def search_growing_emoji(
    batch_size: int, hidden_channels: int, gpu: bool, gpu_index: int
):
    logging.basicConfig(level=logging.INFO)

    """Main function to run the "growing emoji" example task.

    Args:
        batch_size (int, optional): _description_.
        hidden_channels (int, optional): _description_.
    """
    # Display prologue
    NCALab_banner()
    print_mascot(
        "Hello, stranger!\n"
        "I am Bart, the lab rat!\n"
        "I am pleased to hear that you are\n"
        "about to run a hyperparameter search.\n"
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
        #fire_rate=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        #learned_filters=[0, 2],
        dx_noise=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
        num_image_channels=4,
        num_hidden_channels=hidden_channels,
    )
    # No need to search trainer parameters, but we could do that
    trainer_params = ParameterSet(max_epochs=5000)

    # Set up hyperparameter search (grid search)
    search = ParameterSearch(device, GrowingNCAModel, model_params, trainer_params)

    # Print search metadata
    print(search.info())
    print()

    # Run the search!
    search(dataloader_train)


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
