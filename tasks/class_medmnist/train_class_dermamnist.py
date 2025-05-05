#!/usr/bin/env python3
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import (
    ClassificationNCAModel,
    BasicNCATrainer,
    WEIGHTS_PATH,
    show_batch_classification,
    get_compute_device,
)

import numpy as np
import click

from medmnist import DermaMNIST  # type: ignore[import-untyped]

import torch  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]


def train_class_dermamnist(
    batch_size: int,
    hidden_channels: int,
    gpu: bool,
    gpu_index: int,
):
    gradient_clipping = False
    pad_noise = False
    alive_mask = False

    writer = SummaryWriter(
        comment=f"c.hidden_{hidden_channels}_gc_{gradient_clipping}_noise_{pad_noise}_AM_{alive_mask}"
    )

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    T = transforms.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.ConvertImageDtype(dtype=torch.float32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )

    # training dataloader
    dataset_train = DermaMNIST(split="train", download=True, transform=T)
    y_train = dataset_train.labels.squeeze()
    class_sample_count_train = np.array(
        [
            len(np.where(dataset_train.labels == t)[0])
            for t in np.sort(np.unique(y_train))
        ]
    )
    weight = 1.0 / class_sample_count_train
    sample_weight_train = np.array([weight[t] for t in y_train])
    sampler_train = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(sample_weight_train),
        num_samples=len(dataset_train),
        replacement=True,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=batch_size
    )

    # validation dataloader
    dataset_val = DermaMNIST(split="val", download=True, transform=T)
    y_val = dataset_val.labels.squeeze()
    class_sample_count_val = np.array(
        [len(np.where(dataset_val.labels == t)[0]) for t in np.sort(np.unique(y_val))]
    )
    weight = 1.0 / class_sample_count_val
    sample_weight_val = np.array([weight[t] for t in y_val])
    sampler_val = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(sample_weight_val),
        num_samples=len(dataset_val),
        replacement=True,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=32
    )

    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=7,
        use_alive_mask=alive_mask,
        fire_rate=0.5,
        pad_noise=pad_noise,
        filter_padding="circular",
    )
    trainer = BasicNCATrainer(
        nca,
        WEIGHTS_PATH / "classification_dermamnist.pth",
        batch_repeat=2,
        max_epochs=1000,
        gradient_clipping=gradient_clipping,
        steps_range=(64, 96),
        steps_validation=72,
    )
    trainer.train(
        loader_train,
        loader_val,
        summary_writer=writer,
        plot_function=show_batch_classification,
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
def main(batch_size, hidden_channels, gpu: bool, gpu_index: int):
    train_class_dermamnist(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
