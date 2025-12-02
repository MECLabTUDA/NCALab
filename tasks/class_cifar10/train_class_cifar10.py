#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import click
import numpy as np  # type: ignore[import-untyped]
import torch  # type: ignore[import-untyped]
import torchvision  # type: ignore[import-untyped]
from torch.utils.data.sampler import SubsetRandomSampler  # type: ignore[import-untyped]
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from ncalab import (  # noqa: E402
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
default_hidden_channels = 24


def train_class_cifar10(
    batch_size: int,
    hidden_channels: int,
    gpu: bool,
    gpu_index: int,
    max_epochs: int,
    dry: bool,
):
    comment = "CIFAR10"
    comment += f"_hidden_{hidden_channels}"
    comment += f"_gc_{gradient_clipping}"
    comment += f"_noise_{pad_noise}"
    comment += f"_AM_{alive_mask}"
    comment += f"_TE_{use_temporal_encoding}"

    writer = SummaryWriter(comment=comment) if not dry else None

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    # Data loading and preprocessing
    T_train = transforms.Compose(
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
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    T_val = transforms.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.ConvertImageDtype(dtype=torch.float32),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Split train dataset into train and validation
    train_dataset = torchvision.datasets.CIFAR10(
        root=TASK_PATH / "data",
        train=True,
        download=True,
        transform=T_train,
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=TASK_PATH / "data",
        train=True,
        download=True,
        transform=T_val,
    )
    indices = list(range(len(train_dataset)))
    split = int(np.floor(0.1 * len(train_dataset)))
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        sampler=train_sampler,
        pin_memory=True,
    )
    loader_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=len(val_idx),
        num_workers=2,
        sampler=val_sampler,
        pin_memory=True,
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

    # Create NCA model for classification
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
        training_timesteps=(32, 48),
        inference_timesteps=42,
    )
    # Train the NCA model
    trainer = BasicNCATrainer(
        nca,
        WEIGHTS_PATH / "classification_cifar10" if not dry else None,
        batch_repeat=2,
        max_epochs=max_epochs,
        gradient_clipping=gradient_clipping,
    )
    trainer.train(
        loader_train,
        loader_val,
        summary_writer=writer,
    )
    if writer is not None:
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
@click.option("--max-epochs", "-E", type=int, default=120)
@click.option("--dry", "-D", is_flag=True)
def main(
    batch_size, hidden_channels, gpu: bool, gpu_index: int, max_epochs: int, dry: bool
):
    train_class_cifar10(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
        max_epochs=max_epochs,
        dry=dry,
    )


if __name__ == "__main__":
    main()
