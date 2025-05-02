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

import click

from medmnist import BloodMNIST  # type: ignore[import-untyped]

import torch  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]


def train_class_bloodmnist(
    batch_size: int,
    hidden_channels: int,
    gpu: bool,
    gpu_index: int,
    lambda_activity: float,
):
    gradient_clipping = False
    pad_noise = True
    alive_mask = False

    writer = SummaryWriter(
        comment=f"L.act_{lambda_activity}_c.hidden_{hidden_channels}_gc_{gradient_clipping}_noise_{pad_noise}_AM_{alive_mask}"
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

    dataset_train = BloodMNIST(split="train", download=True, transform=T)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True, batch_size=batch_size, drop_last=True
    )

    dataset_val = BloodMNIST(split="val", download=True, transform=T)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=True, batch_size=32, drop_last=True
    )

    print(dataset_train.labels[0])
    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=dataset_train.labels.shape[1],
        hidden_size=128,
        use_alive_mask=alive_mask,
        lambda_activity=lambda_activity,
        fire_rate=0.5,
        pad_noise=pad_noise,
    )
    trainer = BasicNCATrainer(
        nca,
        WEIGHTS_PATH / "selfclass_bloodmnist.pth",
        batch_repeat=2,
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
@click.option("--hidden-channels", "-H", default=9, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
@click.option("--lambda-activity", type=float, default=0.0)
def main(batch_size, hidden_channels, gpu, gpu_index, lambda_activity):
    train_class_bloodmnist(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
        lambda_activity=lambda_activity,
    )


if __name__ == "__main__":
    main()
