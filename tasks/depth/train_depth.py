#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from typing import Any  # type hint
from pathlib import Path, PosixPath  # type hint

from ncalab import (
    DepthNCAModel,
    BasicNCATrainer,
    WEIGHTS_PATH,
    get_compute_device,
)

import click

import torch
import numpy as np

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image

from config import KID_DATASET_PATH


TASK_PATH = Path(__file__).parent


class KIDDataset(Dataset):
    def __init__(self, path: Path | PosixPath, filenames, transform=None) -> None:
        super().__init__()
        self.path = path
        self.image_filenames = filenames
        self.transform = transform
        self.vignette = np.asarray(Image.open(TASK_PATH / "vignette_kid2.png"))[..., 0]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index) -> Any:
        filename = self.image_filenames[index]
        image_filename = KID_DATASET_PATH / "all" / filename
        mask_filename = KID_DATASET_PATH / "depth" / filename
        image = Image.open(image_filename).convert("RGB")
        mask = Image.open(mask_filename).convert("L")
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
        image_arr[self.vignette == 0] = 0
        mask_arr[self.vignette == 0] = 0
        sample = {"image": image_arr, "mask": mask_arr}
        if self.transform is not None:
            sample = self.transform(**sample)
        return sample["image"], sample["mask"]


def train_depth_KID(batch_size: int, hidden_channels: int):
    writer = SummaryWriter()

    device = get_compute_device("cuda:0")

    nca = DepthNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=1,
        lambda_activity=0.00,
    )

    T = A.Compose(
        [
            A.CenterCrop(320, 320),
            A.Resize(80, 80),
            A.RandomRotate90(),
            ToTensorV2(),
        ]
    )
    import pandas as pd

    split = pd.read_csv(TASK_PATH / "split_normal_small_bowel.csv")
    train_filenames = split[split.split != "val"].filename.values
    train_filenames = [
        filename
        for filename in train_filenames
        if (KID_DATASET_PATH / "depth" / filename).exists()
    ]
    train_dataset = KIDDataset(
        KID_DATASET_PATH,
        filenames=train_filenames,
        transform=T,
    )
    val_filenames = split[split.split == "val"].filename.values
    val_filenames = [
        filename
        for filename in val_filenames
        if (KID_DATASET_PATH / "depth" / filename).exists()
    ]
    val_dataset = KIDDataset(
        KID_DATASET_PATH,
        filenames=val_filenames,
        transform=T,
    )

    loader_train = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )
    loader_val = torch.utils.data.DataLoader(
        val_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )

    trainer = BasicNCATrainer(
        nca,
        WEIGHTS_PATH / "depth_KID2_normal_small_bowel.pth",
        max_epochs=500,
        pad_noise=False,
        steps_range=(64, 96),
        steps_validation=80,
    )
    trainer.train_basic_nca(
        loader_train,
        loader_val,
        summary_writer=writer,
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=18, type=int)
def main(batch_size, hidden_channels):
    train_depth_KID(batch_size=batch_size, hidden_channels=hidden_channels)


if __name__ == "__main__":
    main()
