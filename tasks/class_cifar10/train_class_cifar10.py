#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import click
import torch  # type: ignore[import-untyped]
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]
import torchvision  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from ncalab import (
    BasicNCATrainer,
    ClassificationNCAModel,
    get_compute_device,
)

TASK_PATH = Path(__file__).parent.resolve()
WEIGHTS_PATH = TASK_PATH / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)

gradient_clipping = False
pad_noise = False
alive_mask = False
use_temporal_encoding = True
fire_rate = 0.8
default_hidden_channels = 20


def train_class_cifar10(
    batch_size: int,
    hidden_channels: int,
    gpu: bool,
    gpu_index: int,
):
    comment = "CIFAR10"
    comment += f"_hidden_{hidden_channels}"
    comment += f"_gc_{gradient_clipping}"
    comment += f"_noise_{pad_noise}"
    comment += f"_AM_{alive_mask}"
    comment += f"_TE_{use_temporal_encoding}"

    writer = SummaryWriter(comment=comment)

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    # Data loading and preprocessing
    T = transforms.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.ConvertImageDtype(dtype=torch.float32),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=TASK_PATH / "data", train=True, download=True, transform=T
    )
    val_size = int(0.2 * len(trainset))
    train_size = len(trainset) - val_size

    train_subset, val_subset = torch.utils.data.random_split(
        trainset, [train_size, val_size]
    )

    loader_train = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    loader_val = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    class_names = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    num_classes = len(class_names)
    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=num_classes,
        use_alive_mask=alive_mask,
        fire_rate=fire_rate,
        pad_noise=pad_noise,
        use_temporal_encoding=use_temporal_encoding,
        class_names=class_names,
    )
    trainer = BasicNCATrainer(
        nca,
        WEIGHTS_PATH / "classification_cifar10",
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
    train_class_cifar10(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
