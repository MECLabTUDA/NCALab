#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from pathlib import Path, PosixPath
from typing import Any

from ncalab import (
    SegmentationNCAModel,
    CascadeNCA,
    BasicNCATrainer,
    WEIGHTS_PATH,
    get_compute_device,
    print_mascot,
)

from download_kvasir_seg import download_and_extract, KVASIR_SEG_PATH  # type: ignore[import-untyped]


import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]
import click
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, Subset


class KvasirSegDataset(Dataset):
    def __init__(self, path: Path | PosixPath, transform=None) -> None:
        super().__init__()
        self.path = path
        self.image_filenames = sorted((path / "Kvasir-SEG" / "images").glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index) -> Any:
        filename = self.image_filenames[index].name
        image_filename = (self.path / "Kvasir-SEG" / "images" / filename).resolve()
        mask_filename = (self.path / "Kvasir-SEG" / "masks" / filename).resolve()
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


def train_segmentation_kvasir_seg(batch_size: int, hidden_channels: int):
    writer = SummaryWriter()

    device = get_compute_device("cuda:0")

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

    trainer = BasicNCATrainer(cascade, WEIGHTS_PATH / "segmentation_kvasir_seg.pth")
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
def main(batch_size, hidden_channels):
    print_mascot(
        "So you decided to run NCAs on a medical dataset.\n"
        "\n"
        "Kvasir-SEG: That's an endoscopic dataset for polyp segmentation.\n"
        "I'm excited to see how NCAs will perform on that."
    )

    if not KVASIR_SEG_PATH.exists():
        print_mascot(
            "I could not find the Kvasir-SEG dataset on your device.\n"
            "Let me download it for you, it might take a minute."
        )
        download_and_extract()

    train_segmentation_kvasir_seg(
        batch_size=batch_size, hidden_channels=hidden_channels
    )


if __name__ == "__main__":
    main()
