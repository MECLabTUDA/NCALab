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
    fix_random_seed,
)

import click

import cv2

import numpy as np
import pandas as pd

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image

from config import KID_DATASET_PATH, KVASIR_CAPSULE_DATASET_PATH


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
        image_filename = self.path / "all" / filename
        mask_filename = self.path / "depth" / filename
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


class KvasirCapsuleDataset(Dataset):
    def __init__(self, path: Path | PosixPath, filenames, transform=None) -> None:
        super().__init__()
        self.path = path
        self.image_filenames = filenames
        self.transform = transform
        self.vignette = cv2.imread(str(path / "vignette_kvasir_capsule.png"))[..., 0]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image_filename = self.path / "images" / "Any" / filename
        mask_filename = self.path / "depth" / filename
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


class EndoSLAMDataset(Dataset):
    def __init__(self, path: Path | PosixPath, filenames, transform=None) -> None:
        super().__init__()
        self.path = path
        self.image_filenames = filenames
        self.transform = transform
        self.vignette = np.asarray(Image.open(path / "vignette_unity.png"))[..., 0]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index) -> Any:
        filename = self.image_filenames[index]
        image_filename = self.path / "Frames" / filename
        mask_filename = self.path / "Pixelwise Depths" / ("aov_" + filename)
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
    fix_random_seed()
    device = get_compute_device("cuda:0")

    nca = DepthNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=1,
        pad_noise=True,
    )

    INPUT_SIZE = 64

    T = A.Compose(
        [
            A.CenterCrop(300, 300),
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.RandomRotate90(),
            ToTensorV2(),
        ]
    )
    T_val = A.Compose(
        [
            A.CenterCrop(300, 300),
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.RandomRotate90(),
            ToTensorV2(),
        ]
    )

    dataset_id = "kvasircapsule"

    if dataset_id == "kid":
        split = pd.read_csv(KID_DATASET_PATH / "split_depth.csv")
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
            transform=T_val,
        )
    elif dataset_id == "kvasircapsule":
        split = pd.read_csv(KVASIR_CAPSULE_DATASET_PATH / "split_depth.csv")
        train_filenames = split[split.split != "val"].filename.values
        train_filenames = [
            filename
            for filename in train_filenames
            if (KVASIR_CAPSULE_DATASET_PATH / "depth" / filename).exists()
        ]
        train_dataset = KvasirCapsuleDataset(
            KVASIR_CAPSULE_DATASET_PATH,
            filenames=train_filenames,
            transform=T,
        )
        val_filenames = split[split.split == "val"].filename.values
        val_filenames = [
            filename
            for filename in val_filenames
            if (KVASIR_CAPSULE_DATASET_PATH / "depth" / filename).exists()
        ]
        val_dataset = KvasirCapsuleDataset(
            KVASIR_CAPSULE_DATASET_PATH,
            filenames=val_filenames,
            transform=T_val,
        )
    elif dataset_id == "endoslam":
        endoslam_path = Path("~/EndoSLAM/data").expanduser()
        filenames = [
            f.name
            for i, f in enumerate(sorted((endoslam_path / "Frames").glob("*.png")))
            if i % 100 == 0
        ]
        train_filenames = filenames[: int(len(filenames) * 0.8)]
        val_filenames = filenames[len(train_filenames) :]
        train_dataset = EndoSLAMDataset(
            endoslam_path,
            train_filenames,
            transform=T,
        )
        val_dataset = EndoSLAMDataset(
            endoslam_path,
            val_filenames,
            transform=T,
        )

    loader_train = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )
    loader_val = torch.utils.data.DataLoader(
        val_dataset, shuffle=True, batch_size=batch_size, drop_last=True
    )
    nca.vignette = train_dataset.vignette

    trainer = BasicNCATrainer(
        nca,
        WEIGHTS_PATH / f"depth_{dataset_id}.pth",
        max_epochs=2000,
        steps_range=(96, 110),
        steps_validation=100,
    )
    trainer.train(
        loader_train,
        loader_val,
        summary_writer=writer,
        save_every=1,
    )
    writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=20, type=int)
def main(batch_size, hidden_channels):
    train_depth_KID(batch_size=batch_size, hidden_channels=hidden_channels)


if __name__ == "__main__":
    main()
