#!/usr/bin/env python3
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from pathlib import Path


from ncalab import (
    SegmentationNCAModel,
    CascadeNCA,
    BasicNCATrainer,
    get_compute_device,
    print_mascot,
    print_NCALab_banner,
    fix_random_seed,
)

from download_kvasir_seg import download_and_extract, KVASIR_SEG_PATH  # type: ignore[import-untyped]
from dataset_kvasir_seg import KvasirSegDataset

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]
import click

from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset


TASK_PATH = Path(__file__).parent.resolve()
WEIGHTS_PATH = TASK_PATH / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)


def train_segmentation_kvasir_seg(
    batch_size: int, hidden_channels: int, gpu: bool, gpu_index: int
):
    writer = SummaryWriter(comment="Segmentation Kvasir-SEG")
    print_NCALab_banner()
    fix_random_seed()
    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    nca = SegmentationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=1,
        pad_noise=True,
        fire_rate=0.8,
    )
    cascade = CascadeNCA(nca, [8, 4, 2, 1], [70, 20, 10, 5])

    T = A.Compose(
        [
            A.RandomCrop(300, 300),
            A.Resize(256, 256),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            ToTensorV2(),
        ]
    )
    dataset = KvasirSegDataset(KVASIR_SEG_PATH, transform=T)

    train_indices, val_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.image_filenames,
        test_size=int(len(dataset) * 0.2),
    )

    train_split = Subset(dataset, train_indices)
    val_split = Subset(dataset, val_indices)

    loader_train = torch.utils.data.DataLoader(
        train_split, shuffle=True, batch_size=batch_size, drop_last=True
    )
    loader_val = torch.utils.data.DataLoader(
        val_split, shuffle=True, batch_size=batch_size, drop_last=True
    )

    trainer = BasicNCATrainer(cascade, WEIGHTS_PATH / "segmentation_kvasir_seg")
    trainer.train(
        loader_train,
        loader_val,
        summary_writer=writer,
        save_every=1,
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=18, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def main(batch_size, hidden_channels, gpu, gpu_index):
    print_mascot(
        "You're training NCAs on a medical dataset now.\n"
        "\n"
        "Kvasir-SEG is an endoscopic dataset for polyp segmentation."
    )

    if not KVASIR_SEG_PATH.exists():
        print_mascot(
            "I could not find the Kvasir-SEG dataset on your device.\n"
            "Let me download it for you, it might take a minute."
        )
        download_and_extract()

    train_segmentation_kvasir_seg(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
