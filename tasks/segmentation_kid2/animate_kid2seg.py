#!/usr/bin/env python3
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from pathlib import Path

from ncalab import SegmentationNCAModel, CascadeNCA, WEIGHTS_PATH, get_compute_device, fix_random_seed, pad_input

import click

import torch

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.animation as animation  # type: ignore[import-untyped]
import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

from config import (
    KID_SEGMENTATION_MODEL_NAME,
    KID_DATASET_PATH_NNUNET,
)
from kid2dataset import KIDDataset
from baselines import *

from torch.utils.data import DataLoader

import numpy as np

TASK_PATH = Path(__file__).parent


@click.command()
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
@click.option("--hidden-channels", "-H", default=18, type=int)
@click.option(
    "--folds",
    "-f",
    help="Number of folds for k-fold cross validation",
    default=5,
    type=int,
)
@click.option("--id", "-i", help="nnUNet dataset ID", type=int, default=11)
def animate_kid2seg(gpu: bool, gpu_index: int, hidden_channels, folds: int, id: int):
    fix_random_seed()

    T = A.Compose(
        [
            A.CenterCrop(320, 320),
            ToTensorV2(),
        ]
    )

    dataset_test = KIDDataset(
        KID_DATASET_PATH_NNUNET
        / "nnUNet_raw"
        / f"Dataset{id:03d}_KID2vascular",
        Set="Ts",
        transform=T,
    )
    dataloader_test = DataLoader(dataset_test, batch_size=1)

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    models = []
    for fold in range(folds):
        nca = SegmentationNCAModel(
            device,
            num_image_channels=3,
            num_hidden_channels=hidden_channels,
            num_classes=1,
        )
        cascade = CascadeNCA(nca, [8, 4, 2, 1], [50, 25, 15, 15])
        cascade.load_state_dict(
            torch.load(
                WEIGHTS_PATH / f"{KID_SEGMENTATION_MODEL_NAME}_fold{fold:02d}.best.pth",
                weights_only=True,
            )
        )
        cascade.eval()
        models.append(cascade)

    images = np.array([])
    background = None
    with torch.no_grad():
        for i, sample in enumerate(iter(dataloader_test)):
            x, y = sample["image"], sample["mask"]
            x = x.to(device)
            y = y.to(device)
            x = pad_input(x, models[0], noise=True)
            x = models[0].prepare_input(x)

            ensemble_images = []
            for model in models:
                pred = model.record_steps(x)
                mask = [p[0, :, :, -1].cpu().numpy() for p in pred]
                ensemble_images.append(mask)
            images = np.median(np.array(ensemble_images), axis=0)
            images = (images > 0.01).astype(np.int32)

            background = x[0, :3, :, :].permute(1, 2, 0).cpu().numpy()
            if i == 5:
                break

    fig, ax = plt.subplots()
    fig.set_size_inches(2, 2)
    ax.imshow(background)
    im = ax.imshow(images[0], animated=True, alpha=0.5)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.tight_layout()

    def update(i):
        im.set_array(images[i])
        return (im,)

    animation_fig = animation.FuncAnimation(
        fig,
        update,
        frames=len(images),
        interval=100,
        blit=True,
        repeat=True,
        repeat_delay=3000,
    )
    animation_fig.save(TASK_PATH / "figures/segmentation_kid2.gif")


if __name__ == "__main__":
    animate_kid2seg()
