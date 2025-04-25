#!/usr/bin/env python3
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

import logging
from pathlib import Path  # type hint

from ncalab import (
    SegmentationNCAModel,
    BasicNCATrainer,
    WEIGHTS_PATH,
    get_compute_device,
    NCALab_banner,
    print_mascot,
    fix_random_seed,
    CascadeNCA,
    KFoldCrossValidationTrainer,
    SplitDefinition,
)

import click

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

from config import (
    KID_DATASET_PATH,
    KID_SEGMENTATION_MODEL_NAME,
    KID_DATASET_PATH_NNUNET,
)
from kid2dataset import KIDDataset

TASK_PATH = Path(__file__).parent


def train_segmentation_KID(batch_size: int, hidden_channels: int, folds: int, dataset_id: int):
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
        "KID 2 is one of very few capsule endoscopy dataset that features segmentation masks\n"
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
    cascade = CascadeNCA(nca, [8, 4, 2, 1], [50, 25, 15, 15])

    T = A.Compose(
        [
            A.CenterCrop(320, 320),
            A.RandomRotate90(),
            ToTensorV2(),
        ]
    )

    split = SplitDefinition.read(
        KID_DATASET_PATH_NNUNET
        / "nnUNet_preprocessed" / f"Dataset{dataset_id:03d}_KID2vascular" / "splits_final.json"
    )

    trainer = BasicNCATrainer(
        cascade, WEIGHTS_PATH / f"{KID_SEGMENTATION_MODEL_NAME}.pth", max_epochs=1000
    )
    kfold = KFoldCrossValidationTrainer(trainer, split)

    summaries = kfold.train(
        KIDDataset,
        KID_DATASET_PATH_NNUNET / "nnUNet_raw" / f"Dataset{dataset_id:03d}_KID2vascular",
        T,
        {"train": batch_size, "val": batch_size},
        save_every=1,
    )
    for fold, summary in enumerate(summaries):
        df = summary.to_dataframe()
        print(df)
        print(
            f"Dice: {df.Dice.mean():.4f} +- {df.Dice.std():.4f}  IoU: {df.IoU.mean():.4f} +- {df.IoU.std():.4f}"
        )


@click.command()
@click.option("--batch-size", "-b", default=4, type=int)
@click.option("--hidden-channels", "-H", default=18, type=int)
@click.option(
    "--folds",
    "-f",
    help="Number of folds for k-fold cross validation",
    default=1,
    type=int,
)
@click.option("--id", "-i", help="nnUNet dataset ID", type=int, default=11)
def main(batch_size, hidden_channels, folds, id):
    train_segmentation_KID(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        folds=folds,
        dataset_id=id,
    )


if __name__ == "__main__":
    main()
