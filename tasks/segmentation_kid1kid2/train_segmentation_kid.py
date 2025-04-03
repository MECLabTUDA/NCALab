#!/usr/bin/env python3
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

import logging
from pathlib import Path, PosixPath  # type hint
from typing import Any  # type hint

from ncalab import (
    SegmentationNCAModel,
    BasicNCATrainer,
    WEIGHTS_PATH,
    get_compute_device,
    NCALab_banner,
    print_mascot,
    fix_random_seed,
    generate_Kfold_idx,
)

import click

import numpy as np

import pandas as pd

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

import torch
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
        vignette_path = TASK_PATH / "vignette_kid2.png"
        self.vignette = np.asarray(Image.open(vignette_path))[..., 0]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index) -> Any:
        filename = self.image_filenames[index]
        image_filename = self.path / "vascular" / filename
        mask_filename = filename[: -len(".png")] + "m" + ".png"
        mask_filename = self.path / "vascular-annotations" / mask_filename
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


def train_segmentation_KID(batch_size: int, hidden_channels: int, folds: int):
    logging.basicConfig(level=logging.INFO)
    NCALab_banner()

    if not KID_DATASET_PATH.exists():
        print_mascot(
            "It seems like you didn't properly download and reference\n"
            "the KID 2 dataset.\n"
            "\n"
            f"Please see {TASK_PATH / 'config.py.example'} for details,\n"
            f"and make sure you copy that file to {TASK_PATH / 'config.py'}\n"
            "with your own settings.\n"
        )
        return

    print_mascot(
        "KID 2 is a capsule endoscopy dataset that features segmentation masks\n"
        "-- which means that we can train a segmentation model on it!\n"
        "In this example, we only use class 'vascular', which shows vascular\n"
        "lesions inside the gastro-intestinal tract, such as angiectasia or\n"
        "bleedings."
    )

    assert folds > 0

    fix_random_seed()
    device = get_compute_device("cuda:0")

    nca = SegmentationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=1,
    )

    T = A.Compose(
        [
            A.RandomCrop(256, 256),
            A.Resize(64, 64),
            A.RandomRotate90(),
            ToTensorV2(),
        ]
    )

    split = pd.read_csv(TASK_PATH / "split_vascular.csv")

    if folds == 1:
        summary_writer = SummaryWriter(comment="kid2seg")

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

        trainer = BasicNCATrainer(
            nca,
            WEIGHTS_PATH / "segmentation_KID2_vascular.pth",
            max_epochs=1000,
            # p_retain_pool=1.0,
        )
        trainer.train(
            loader_train,
            loader_val,
            summary_writer=summary_writer,
        )
    else:
        data = split.filename.values
        all_folds = generate_Kfold_idx(data, folds)
        summaries = []
        for i, fold_idx in enumerate(all_folds):
            logging.info(
                f"Training with k-fold cross validation, fold {i + 1} / {folds}"
            )
            summary_writer = SummaryWriter(comment=f"kid2seg_fold_{i+1}")
            train_idx, val_idx = fold_idx
            filenames_train = data[train_idx]
            filenames_val = data[val_idx]
            train_dataset = KIDDataset(
                KID_DATASET_PATH,
                filenames=filenames_train,
                transform=T,
            )
            val_dataset = KIDDataset(
                KID_DATASET_PATH,
                filenames=filenames_val,
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
                WEIGHTS_PATH / f"segmentation_KID2_vascular_f{i+1}.pth",
                max_epochs=1000,
                # p_retain_pool=1.0,
            )
            summary = trainer.train(
                loader_train,
                dataloader_val=loader_val,
                dataloader_test=loader_val,
                summary_writer=summary_writer,
                save_every=3,
            )
            summaries.append(summary.to_dict())
            summary_writer.close()
        df = pd.DataFrame(summaries)
        print(df)
        print(
            f"Dice: {df.Dice.mean():.4f} +- {df.Dice.std():.4f}  IoU: {df.IoU.mean():.4f} +- {df.IoU.std():.4f}"
        )


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=16, type=int)
@click.option(
    "--folds",
    "-f",
    help="Number of folds for k-fold cross validation",
    default=1,
    type=int,
)
def main(batch_size, hidden_channels, folds):
    train_segmentation_KID(
        batch_size=batch_size, hidden_channels=hidden_channels, folds=folds
    )


if __name__ == "__main__":
    main()
