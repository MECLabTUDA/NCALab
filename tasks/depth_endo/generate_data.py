#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

import shutil
from pathlib import Path, PosixPath  # for type hints
import random

from ncalab import get_compute_device, fix_random_seed
from config import KID_DATASET_PATH, KVASIR_CAPSULE_DATASET_PATH

import cv2
import click
from transformers import pipeline
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class KIDDataset(Dataset):
    def __init__(self, path: Path | PosixPath, transform=None) -> None:
        super().__init__()
        self.path = path
        self.image_filenames = os.listdir(path / "all")
        self.transform = transform
        self.vignette = cv2.imread(str(path / "vignette_kid2.png"))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image_filename = self.path / "all" / filename
        image = Image.open(image_filename).convert("RGB")
        sample = {"image": image}
        if self.transform is not None:
            sample = self.transform(**sample)
        return sample["image"], filename


class KvasirCapsuleDataset(Dataset):
    def __init__(self, path: Path | PosixPath, transform=None) -> None:
        super().__init__()
        self.path = path
        self.image_filenames = os.listdir(path / "images" / "Any")
        self.transform = transform
        self.vignette = cv2.imread(str(path / "vignette_kvasir_capsule.png"))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image_filename = self.path / "images" / "Any" / filename
        image = Image.open(image_filename).convert("RGB")
        sample = {"image": image}
        if self.transform is not None:
            sample = self.transform(**sample)
        return sample["image"], filename


def masked_gradient_magnitude(depth, vignette):
    depth_np = np.asarray(depth)
    depth_np = np.uint8(
        (
            (depth_np - np.min(depth_np[vignette[..., 0] != 0]))
            / (
                np.max(depth_np[vignette[..., 0] != 0])
                - np.min(depth_np[vignette[..., 0] != 0])
            )
        )
        * 255.0
    )
    depth_np[vignette[..., 0] == 0] = 0

    g_x = np.gradient(depth_np, axis=0)
    g_y = np.gradient(depth_np, axis=1)
    return depth_np, np.sum(np.sqrt(g_x**2, g_y**2)) / (
        depth_np.shape[0] * depth_np.shape[1]
    )


@click.command()
@click.option("-f", "--filter-flat-maps", is_flag=True, default=True)
def main(filter_flat_maps):
    fix_random_seed()
    device = get_compute_device()

    pipe_small = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device,
    )
    pipe_large = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=device,
    )

    # Create directory "Any" containing images of all classes
    KVASIR_CAPSULE_IMAGES_PATH = KVASIR_CAPSULE_DATASET_PATH / "images"
    (KVASIR_CAPSULE_IMAGES_PATH / "Any").mkdir(exist_ok=True)
    for subdir in KVASIR_CAPSULE_IMAGES_PATH.glob("*"):
        if subdir.name == "Any":
            continue
        for path in subdir.glob("*.jpg"):
            shutil.copy2(path, KVASIR_CAPSULE_IMAGES_PATH / "Any")

    dataset = KvasirCapsuleDataset(KVASIR_CAPSULE_DATASET_PATH)

    # create split definition file
    filenames = {}
    patient_set = []
    click.secho("Creating data split definition.", fg="blue")
    for image_filename in tqdm(sorted(os.listdir(dataset.path / "depth"))):
        patient = image_filename.split("_")[0]
        if patient not in patient_set:
            patient_set.append(patient)
            filenames[patient] = []
        filenames[patient].append(image_filename)

    X_train, X_test = train_test_split(patient_set, test_size=0.1, random_state=1337)
    X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=1337)

    # create by-patient train/val/test split
    with open(dataset.path / "split_depth.csv", "w") as f:
        f.write("filename,split" + os.linesep)
        for patient, image_filenames in tqdm(filenames.items()):
            if patient in X_train:
                Set = "train"
            elif patient in X_val:
                Set = "val"
            elif patient in X_test:
                Set = "test"
            # shuffle list of filenames and reduce its length,
            # to avoid excessive number of redundant samples
            # --> there are more intelligent ways to do this, but it is something
            image_filenames = random.sample(image_filenames, len(image_filenames))[:50]
            for image_filename in image_filenames:
                f.write(f"{image_filename},{Set}" + os.linesep)

    # KID2 Dataset (bleeding)
    dataset = KIDDataset(KID_DATASET_PATH)
    output_path = Path(dataset.path / "depth")
    output_path.mkdir(exist_ok=True)
    for sample in tqdm(dataset):
        image, filename = sample
        depth_s = pipe_small(image)["depth"]
        depth_np, score = masked_gradient_magnitude(depth_s, dataset.vignette)
        if filter_flat_maps and score < 1.1:
            depth_l = pipe_large(image)["depth"]
            print("Resorted to larger model.")
            depth_np, score = masked_gradient_magnitude(depth_l, dataset.vignette)
        if score >= 1.1 or not filter_flat_maps:
            cv2.imwrite(str(output_path / filename), depth_np)
    X_train, X_test = train_test_split(filenames, test_size=0.1, random_state=1337)
    X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=1337)
    with open(dataset.path / "split_depth.csv", "w") as f:
        f.write("filename,split" + os.linesep)
        for x in X_train:
            f.write(f"{x},train" + os.linesep)
        for x in X_val:
            f.write(f"{x},val" + os.linesep)
        for x in X_test:
            f.write(f"{x},test" + os.linesep)


if __name__ == "__main__":
    main()
