#!/usr/bin/env python3
import pytest

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

from torch.utils.data import Dataset, DataLoader

import numpy as np

from ncalab import (
    DepthNCAModel,
    CascadeNCA,
    BasicNCATrainer,
    get_compute_device,
)


T = A.Compose(
    [
        ToTensorV2(),
    ]
)


class DummyDepthDataset(Dataset):
    def __init__(self, transform=None) -> None:
        super().__init__()
        self.images = np.ones((16, 32, 32, 3), dtype=np.float32)
        self.masks = np.ones((16, 32, 32), dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        sample = {"image": self.images[index], "mask": self.masks[index]}
        if self.transform is not None:
            sample = self.transform(**sample)
        return sample["image"], sample["mask"]


def test_cascade_depth_training_with_validation():
    """
    Test if a basic NCA trainer runs through for a few epochs without exception.
    """
    device = get_compute_device("cpu")

    nca = DepthNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=5,
        pad_noise=False,
    )

    cascade = CascadeNCA(nca, [2, 1], [3, 3])

    dataset = DummyDepthDataset(transform=T)
    dataloader_train = DataLoader(dataset, batch_size=8, shuffle=False)
    dataloader_val = DataLoader(dataset, batch_size=8, shuffle=False)

    try:
        trainer = BasicNCATrainer(cascade, None, max_epochs=3)
    except Exception as e:
        pytest.fail(e)

    try:
        trainer.train(dataloader_train, dataloader_val, save_every=100)
    except Exception as e:
        pytest.fail(e)
