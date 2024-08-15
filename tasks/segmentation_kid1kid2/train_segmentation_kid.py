#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from typing import Any
from pathlib import Path, PosixPath

from nca import SegmentationNCAModel, BasicNCATrainer, WEIGHTS_PATH, get_compute_device

import click

import torch
import numpy as np

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, Subset
from PIL import Image

from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from config import KID_DATASET_PATH


class KIDDataset(Dataset):
    def __init__(self, path: Path | PosixPath, filenames, transform=None) -> None:
        super().__init__()
        self.path = path
        self.image_filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index) -> Any:
        filename = self.image_filenames[index]
        image_filename = self.path / "vascular" / filename
        mask_filename = filename[: -len(".png")] + "m" + ".png"
        mask_filename = self.path / "vascular-annotations" / mask_filename
        image = Image.open(image_filename).convert("RGB")
        mask = Image.open(mask_filename).convert("L")
        bbox = image.getbbox()
        image = image.crop(bbox)
        mask = mask.crop(bbox)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
        sample = {"image": image_arr, "mask": mask_arr}
        sample = self.transform(**sample)
        return sample["image"], sample["mask"]


def train_segmentation_KID(batch_size: int, hidden_channels: int):
    writer = SummaryWriter()

    device = get_compute_device("cuda:0")

    nca = SegmentationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=1,
        lambda_activity=0.00
    )

    T = A.Compose(
        [
            A.RandomCrop(256, 256),
            A.Resize(64, 64),
            A.Flip(),
            ToTensorV2(),
        ]
    )
    import pandas as pd
    split = pd.read_csv(KID_DATASET_PATH / "split_vascular.csv")
    train_dataset = KIDDataset(
        KID_DATASET_PATH,
        filenames=split[split.split == "train"].filename.values,
        transform=T,
    )
    val_dataset = KIDDataset(
        KID_DATASET_PATH,
        filenames=split[split.split == "val"].filename.values,
        transform=T,
    )

    loader_train = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )
    loader_val = torch.utils.data.DataLoader(
        val_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )

    trainer = BasicNCATrainer(nca, WEIGHTS_PATH / "segmentation_KID2_vascular.pth", max_epochs=1000)
    trainer.train_basic_nca(
        loader_train,
        loader_val,
        summary_writer=writer,
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=16, type=int)
def main(batch_size, hidden_channels):
    train_segmentation_KID(batch_size=batch_size, hidden_channels=hidden_channels)


if __name__ == "__main__":
    main()
