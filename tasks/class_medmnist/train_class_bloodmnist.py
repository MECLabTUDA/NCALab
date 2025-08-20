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
)

import click

from medmnist import INFO, BloodMNIST  # type: ignore[import-untyped]

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


def train_class_bloodmnist(
    batch_size: int,
    hidden_channels: int,
    gpu: bool,
    gpu_index: int,
):
    comment = "BloodMNIST"
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

    dataset_train = BloodMNIST(split="train", download=True, transform=T)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True, batch_size=batch_size, drop_last=True
    )

    dataset_val = BloodMNIST(split="val", download=True, transform=T)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=True, batch_size=32, drop_last=True
    )

    num_classes = len(INFO["bloodmnist"]["label"])
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
        WEIGHTS_PATH / "classification_bloodmnist",
        batch_repeat=2,
        max_epochs=40,
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
    train_class_bloodmnist(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
