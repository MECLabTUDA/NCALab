#!/usr/bin/env python3
from pathlib import Path
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import (
    ClassificationNCAModel,
    BasicNCATrainer,
    get_compute_device,
    print_NCALab_banner,
)

import numpy as np
import click

from medmnist import INFO, DermaMNIST  # type: ignore[import-untyped]

import torch  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]

TASK_PATH = Path(__file__).parent.resolve()
WEIGHTS_PATH = TASK_PATH / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)

gradient_clipping = False
pad_noise = False
alive_mask = False
use_temporal_encoding = True
fire_rate = 0.8
default_hidden_channels = 20


def train_class_dermamnist(
    batch_size: int,
    hidden_channels: int,
    gpu: bool,
    gpu_index: int,
):
    print_NCALab_banner()

    comment = "DermaMNIST"
    comment += f"_hidden_{hidden_channels}"
    comment += f"_gc_{gradient_clipping}"
    comment += f"_noise_{pad_noise}"
    comment += f"_AM_{alive_mask}"
    comment += f"_TE_{use_temporal_encoding}"

    writer = SummaryWriter(comment=comment)

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    T = transforms.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.ConvertImageDtype(dtype=torch.float32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize((0.5,), (0.225,)),
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
    sample_weight_train = [weight[int(t)].astype(float) for t in y_train]
    sampler_train = torch.utils.data.WeightedRandomSampler(
        sample_weight_train,
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
    sample_weight_val = [weight[int(t)].astype(float) for t in y_val]
    sampler_val = torch.utils.data.WeightedRandomSampler(
        sample_weight_val,
        num_samples=len(dataset_val),
        replacement=True,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val, batch_size=len(dataset_val)
    )

    num_classes = len(INFO["dermamnist"]["label"])
    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=num_classes,
        use_alive_mask=alive_mask,
        fire_rate=fire_rate,
        pad_noise=pad_noise,
        use_temporal_encoding=use_temporal_encoding,
    )

    trainer = BasicNCATrainer(
        nca,
        WEIGHTS_PATH / "classification_dermamnist",
        batch_repeat=2,
        max_epochs=500,
        gradient_clipping=gradient_clipping,
        steps_range=(32, 33),
        steps_validation=32,
    )
    trainer.train(
        loader_train,
        loader_val,
        summary_writer=writer,
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=default_hidden_channels, type=int)
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
