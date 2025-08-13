
from pathlib import Path, PosixPath
from typing import Any

import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class KvasirSegDataset(Dataset):
    def __init__(self, path: Path | PosixPath, transform) -> None:
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
