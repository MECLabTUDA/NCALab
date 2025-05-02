#!/usr/bin/env python3
import sys
import os
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import DepthNCAModel, WEIGHTS_PATH, get_compute_device

import click

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import pandas as pd

from PIL import Image

import numpy as np

import torch

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

from transformers import pipeline

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as rmse
from skimage.metrics import peak_signal_noise_ratio as psnr

import cv2


@click.command()
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
@click.option("--hidden-channels", "-H", default=14, type=int)
def eval_depth(gpu: bool, gpu_index: int, hidden_channels: int):
    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    nca = DepthNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=1,
        lambda_activity=0.00,
        pad_noise=False,
    ).to(device)

    nca.load_state_dict(
        torch.load(
            WEIGHTS_PATH / "depth_KID2_normal_small_bowel.best.pth", weights_only=True
        )
    )
    nca.eval()

    frame_path = Path("~/EndoSLAM/data/Frames").expanduser()
    depth_path = Path("~/EndoSLAM/data/Pixelwise Depths").expanduser()

    vignette_path = frame_path / ".." / "vignette_unity.png"
    vignette = cv2.imread(vignette_path)
    vignette = vignette[:, :, 0]

    INPUT_SIZE = 64

    T = A.Compose(
        [
            A.CenterCrop(320, 320),
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            ToTensorV2(),
        ]
    )

    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device="cuda:0",
    )

    scores = {
        "ssim_nca": [],
        "ssim_DAv2": [],
        "rmse_nca": [],
        "rmse_DAv2": [],
        "psnr_nca": [],
        "psnr_DAv2": [],
    }

    for i, frame_image_path in enumerate(frame_path.glob("*")):
        frame_filename = frame_image_path.name
        frame_index = frame_filename.split("_")[1][:-4]
        depth_image_path = depth_path / f"aov_image_{frame_index}.png"

        image = Image.open(frame_image_path)
        gt = np.asarray(Image.open(depth_image_path))[:, :, 0]

        depth_DAv2 = pipe(image)["depth"]
        depth_DAv2 = np.asarray(depth_DAv2)
        depth_DAv2 = depth_DAv2 - np.min(
            depth_DAv2
        )  # / (np.max(depth_DAv2) - np.min(depth_DAv2))

        image = cv2.imread(frame_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = T(image=image)["image"].to(device).unsqueeze(0) / 255.0
        depth_nca = nca.estimate_depth(image, steps=100)
        image = image[0].permute(1, 2, 0)

        depth_nca = depth_nca[0, :, :, 0]
        depth_nca = depth_nca.cpu().numpy() * 255.0
        depth_nca = cv2.resize(
            depth_nca, dsize=(320, 320), interpolation=cv2.INTER_CUBIC
        )
        depth_nca = (depth_nca - np.min(depth_nca[vignette != 0])) / (
            np.max(depth_nca[vignette != 0]) - np.min(depth_nca[vignette != 0])
        )
        # depth_nca = np.uint8(255 * depth_nca)

        gt = gt - np.min(
            gt[vignette != 0]
        )  # / (np.max(gt[vignette != 0]) - np.min(gt[vignette != 0]))
        # gt = np.copy(gt)
        depth_DAv2[vignette == 0] = 0
        depth_nca[vignette == 0] = 0
        gt[vignette == 0] = 0

        ssim_nca = ssim(gt, depth_nca, data_range=gt.max() - gt.min())
        ssim_DAv2 = ssim(gt, depth_DAv2, data_range=gt.max() - gt.min())

        rmse_nca = rmse(gt, depth_nca)
        rmse_DAv2 = rmse(gt, depth_DAv2)

        psnr_nca = psnr(gt, depth_nca, data_range=gt.max() - gt.min())
        psnr_DAv2 = psnr(gt, depth_DAv2, data_range=gt.max() - gt.min())

        scores["ssim_nca"].append(ssim_nca)
        scores["rmse_nca"].append(rmse_nca)
        scores["psnr_nca"].append(psnr_nca)

        scores["ssim_DAv2"].append(ssim_DAv2)
        scores["rmse_DAv2"].append(rmse_DAv2)
        scores["psnr_DAv2"].append(psnr_DAv2)

        s = ""
        for k, v in scores.items():
            if not v:
                continue
            s += f"{k}: {v[-1]:.3f}   "
        print(s)
        print(gt.min(), gt.max())

        fig, ax = plt.subplots(4, 1)
        ax[0].imshow(image.cpu().numpy())
        ax[1].imshow(gt, cmap="magma")
        ax[2].imshow(depth_nca, cmap="magma")
        ax[3].imshow(depth_DAv2, cmap="magma")
        plt.show()
        # break
    df = pd.DataFrame(scores)
    df.to_csv("depth_scores.csv")


if __name__ == "__main__":
    eval_depth()
