import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from typing import Any  # type hint
from pathlib import Path, PosixPath  # type hint

import cv2

import numpy as np

from torch.utils.data import Dataset
from PIL import Image


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
