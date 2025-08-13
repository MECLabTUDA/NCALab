#!/usr/bin/env python3
import os
import sys
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import (
    Animator,
    SegmentationNCAModel,
    CascadeNCA,
    get_compute_device,
    print_NCALab_banner,
    fix_random_seed
)

from download_kvasir_seg import KVASIR_SEG_PATH  # type: ignore[import-untyped]
from dataset_kvasir_seg import KvasirSegDataset

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]
import click

import torch


TASK_PATH = Path(__file__).parent
FIGURE_PATH = TASK_PATH / "figures"
FIGURE_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH = TASK_PATH / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)


@click.command()
@click.option("--hidden-channels", "-H", default=18, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def eval_segmentation_kvasir_seg(hidden_channels: int, gpu: bool, gpu_index: int):
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

    cascade.load_state_dict(
        torch.load(
            WEIGHTS_PATH / "segmentation_kvasir_seg" / "last_model.pth",
            weights_only=True,
        )
    )

    seed = dataset[0][0].unsqueeze(0).to(device)
    animator = Animator(cascade, seed, overlay=True)

    out_path = FIGURE_PATH / "segmentation_kvasir_seg.gif"
    animator.save(out_path)
    click.secho(f"Done. You'll find the generated GIF in {out_path} .")


if __name__ == "__main__":
    eval_segmentation_kvasir_seg()
