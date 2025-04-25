import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from pathlib import Path, PosixPath  # type hint
from typing import Any  # type hint

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

TASK_PATH = Path(__file__).parent


class KIDDataset(Dataset):
    def __init__(
        self, path: Path | PosixPath, image_filenames=None, transform=None, Set="Tr"
    ) -> None:
        super().__init__()
        self.path = path
        assert Set in ("Tr", "Ts")
        self.Set = Set
        if image_filenames is None:
            image_filenames = [
                "_".join(p.name.split("_")[:-1])
                for p in (path / f"images{self.Set}").glob("*.png")
            ]
        self.image_filenames = image_filenames
        self.transform = transform
        vignette_path = TASK_PATH / "vignette_kid2.png"
        self.vignette = np.asarray(Image.open(vignette_path))[..., 0]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index) -> Any:
        filename = self.image_filenames[index]
        image_filename = self.path / f"images{self.Set}" / (filename + "_0000.png")
        mask_filename = self.path / f"labels{self.Set}" / (filename + ".png")
        image = Image.open(image_filename).convert("RGB")
        mask = Image.open(mask_filename).convert("L")
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask, dtype=np.float32)
        if np.max(mask_arr) > 0:
            mask_arr /= np.max(mask_arr)
        image_arr[self.vignette == 0] = 0
        mask_arr[self.vignette == 0] = 0
        sample = {"image": image_arr, "mask": mask_arr, "image_filename": filename}
        if self.transform is not None:
            sample = self.transform(**sample)
        return sample
