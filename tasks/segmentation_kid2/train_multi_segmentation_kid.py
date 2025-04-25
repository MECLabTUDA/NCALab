#!/usr/bin/env python3
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

import logging
from pathlib import Path  # type hint

from ncalab import (
    get_compute_device,
    NCALab_banner,
    print_mascot,
    fix_random_seed,
)

import click

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

from config import KID_DATASET_PATH, KID_SEGMENTATION_MODEL_NAME
from kid2dataset import KIDDataset

TASK_PATH = Path(__file__).parent


import segmentation_models as smp


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

    assert folds > 0

    fix_random_seed()
    device = get_compute_device("cuda:0")

    T = A.Compose(
        [
            A.CenterCrop(320, 320),
            A.RandomRotate90(),
            ToTensorV2(),
        ]
    )

    backbones = [
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "mobilenet_v2",
        "timm-mobilenetv3_small_100",
        "resnet18",
        "resnet34",
        "resnet50",
    ]
    backbone_names_pretty = {
        "efficientnet-b0": "Eff.Net-B0",
        "efficientnet-b1": "Eff.Net-B1",
        "efficientnet-b2": "Eff.Net-B2",
        "mobilenet_v2": "MobileNet2",
        "resnet18": "ResNet18",
        "resnet34": "ResNet34",
        "resnet50": "ResNet50",
        "timm-mobilenetv3_small_100": "MobileNet3",
    }
    for backbone in backbones:
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
        )
        # freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        preprocess_input = smp.encoder.get_preprocessing_fn(
            backbone, pretrained="imagenet"
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
