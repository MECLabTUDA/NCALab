#!/usr/bin/env python3
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

import logging
from pathlib import Path

import click

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ncalab import (
    SegmentationNCAModel,
    WEIGHTS_PATH,
    get_compute_device,
    NCALab_banner,
    print_mascot,
    fix_random_seed,
    CascadeNCA,
    pad_input,
)

from config import (
    KID_DATASET_PATH,
    KID_SEGMENTATION_MODEL_NAME,
    KID_DATASET_PATH_NNUNET,
)
from kid2dataset import KIDDataset
from baselines import *

TASK_PATH = Path(__file__).parent
FIGURE_PATH = TASK_PATH / "figures"
FIGURE_PATH.mkdir(exist_ok=True)


device = get_compute_device("cuda:0")

T = A.Compose(
    [
        A.CenterCrop(320, 320),
        ToTensorV2(),
    ]
)

variances = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]


def boxplot():
    fig, ax = plt.subplots(figsize=(8, 3))
    parts = ax.violinplot(model_accuracy, positions=model_kB, widths=model_kB / 10)

    parts["cbars"].set_alpha(0.0)
    for pc in parts["bodies"]:
        pc.set_facecolor("green")  # Set the face color
        pc.set_edgecolor("green")  # Set the edge color
        pc.set_alpha(0.6)  # Set the transparen
        break
    ax.set_xscale("log")

    for i, label in enumerate(model_names):
        ax.text(
            0.95 * model_kB[i],
            np.min(model_accuracy.T[i]),
            label,
            ha="right",
            va="bottom",
            rotation=90,
        )

    ax.set_xlabel("Model size [kB]".upper(), weight="bold", font="Calibri")
    ax.set_ylabel("Accuracy (Dice)".upper(), weight="bold", font="Calibri")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "accuracy_vs_parameters.pdf", dpi=300)
    plt.show()


def eval_segmentation_KID_baselines(folds: int, dataset_id: int):
    dataset_test = KIDDataset(
        KID_DATASET_PATH_NNUNET
        / "nnUNet_raw"
        / f"Dataset{dataset_id:03d}_KID2vascular",
        Set="Ts",
        transform=T,
    )

    model_sizes = list_trainable_parameters(make_model_zoo())

    lesion_size_vs_dice = {}
    dice_for_variance = {}

    result = []
    for model_name in model_zoo_names:
        lesion_size_vs_dice[model_name] = ([], [])
        dice_for_variance[model_name] = []

        models = []
        for fold in range(folds):
            model = load_model(model_name, fold).to(device)
            models.append(model)

        dataloader_test = DataLoader(dataset_test, batch_size=1)

        with torch.no_grad():
            dice_all = []
            iou_all = []

            for variance in variances:
                for i, sample in enumerate(iter(dataloader_test)):
                    x, y = sample["image"], sample["mask"]
                    lesion_size_vs_dice[model_name][0].append(np.sum(y.numpy()))
                    if variance != 0:
                        x += +(variance**0.5) * torch.randn(*x.shape)
                    x = x.to(device)
                    y = y.to(device)

                    y_pred_ensemble = torch.zeros((folds, *y.shape))
                    for j, model in enumerate(models):
                        y_pred = model(x)
                        y_pred_ensemble[j] = y_pred[0]
                    y_pred_avg = torch.mean(y_pred_ensemble, dim=0).to(device)

                    tp, fp, fn, tn = smp.metrics.get_stats(
                        y_pred_avg.unsqueeze(0),
                        y[:, None, :, :].long(),
                        mode="binary",
                        threshold=0.5,
                    )
                    dice = torch.mean(2.0 * tp / (2.0 * tp + fp + fn)).item()
                    iou = torch.mean(tp / (tp + fp + fn)).item()
                    lesion_size_vs_dice[model_name][1].append(dice)
                    dice_all.append(dice)
                    iou_all.append(iou)
                dice_for_variance[model_name].append(np.mean(dice_all))

        result.append(
            {
                "model_name": model_name,
                "dice_mean": np.mean(dice_all),
                "iou_mean": np.mean(iou_all),
                "dice_std": np.std(dice_all),
                "iou_std": np.std(iou_all),
                "kB": 4 * model_sizes[model_name] / 1000,
            }
        )
    return result, lesion_size_vs_dice, dice_for_variance


def eval_segmentation_KID_NCA(hidden_channels: int, folds: int, dataset_id: int):
    dataset_test = KIDDataset(
        KID_DATASET_PATH_NNUNET
        / "nnUNet_raw"
        / f"Dataset{dataset_id:03d}_KID2vascular",
        Set="Ts",
        transform=T,
    )

    dataloader_test = DataLoader(dataset_test, batch_size=1)

    lesion_size_vs_dice = ([], [])

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

    with torch.no_grad():
        dice_all = []
        iou_all = []
        for i, sample in enumerate(iter(dataloader_test)):
            x, y = sample["image"], sample["mask"]
            lesion_size_vs_dice[0].append(np.sum(y.numpy()))
            x = x.to(device)
            y = y.to(device)
            x = pad_input(x, cascade, noise=True)
            x = cascade.prepare_input(x)
            x = x.permute(0, 2, 3, 1)

            y_pred_ensemble = torch.zeros((folds, *y.shape))
            for j, model in enumerate(models):
                y_pred = model(x)[
                    ..., cascade.num_image_channels + cascade.num_hidden_channels :
                ].permute(0, 3, 1, 2)
                y_pred_ensemble[j] = y_pred[0]
            y_pred_avg = torch.mean(y_pred_ensemble, dim=0).to(device)

            tp, fp, fn, tn = smp.metrics.get_stats(
                y_pred_avg.unsqueeze(0),
                y[:, None, :, :].long(),
                mode="binary",
                threshold=0.5,
            )
            dice = torch.mean(2.0 * tp / (2.0 * tp + fp + fn)).item()
            # if dice < 0.2 and lesion_size_vs_dice[0][-1] > 10000:
            #    sns.set_style("white")
            #    plt.imshow(x.squeeze(0)[..., :3].cpu().numpy())
            #    plt.imshow(y_pred_avg.squeeze(0).cpu().numpy() > 0.5, alpha=0.5)
            #    plt.axis("off")
            #    plt.show()
            lesion_size_vs_dice[1].append(dice)
            iou = torch.mean(tp / (tp + fp + fn)).item()
            dice_all.append(dice)
            iou_all.append(iou)

    model_parameters = filter(lambda p: p.requires_grad, nca.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    result = {
        "model_name": "NCA",
        "dice_mean": np.mean(dice_all),
        "iou_mean": np.mean(iou_all),
        "dice_std": np.std(dice_all),
        "iou_std": np.std(iou_all),
        "kB": 4 * params / 1000,
    }
    return result, lesion_size_vs_dice


def eval_segmentation_KID_NCA_noise(hidden_channels: int, folds: int, dataset_id: int):
    dataset_test = KIDDataset(
        KID_DATASET_PATH_NNUNET
        / "nnUNet_raw"
        / f"Dataset{dataset_id:03d}_KID2vascular",
        Set="Ts",
        transform=T,
    )

    dataloader_test = DataLoader(dataset_test, batch_size=1)

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

    with torch.no_grad():
        dice_all = []
        iou_all = []
        dice_for_variance = []
        for variance in variances:
            for i, sample in enumerate(iter(dataloader_test)):
                x, y = sample["image"], sample["mask"]
                if variance != 0:
                    x += +(variance**0.5) * torch.randn(*x.shape)
                x = x.to(device)
                y = y.to(device)
                x = pad_input(x, cascade, noise=True)
                x = cascade.prepare_input(x)
                x = x.permute(0, 2, 3, 1)

                y_pred_ensemble = torch.zeros((folds, *y.shape))
                for j, model in enumerate(models):
                    y_pred = model(x)[
                        ..., cascade.num_image_channels + cascade.num_hidden_channels :
                    ].permute(0, 3, 1, 2)
                    y_pred_ensemble[j] = y_pred[0]
                y_pred_avg = torch.mean(y_pred_ensemble, dim=0).to(device)

                tp, fp, fn, tn = smp.metrics.get_stats(
                    y_pred_avg.unsqueeze(0),
                    y[:, None, :, :].long(),
                    mode="binary",
                    threshold=0.5,
                )
                dice = torch.mean(2.0 * tp / (2.0 * tp + fp + fn)).item()
                iou = torch.mean(tp / (tp + fp + fn)).item()
                dice_all.append(dice)
                iou_all.append(iou)
            dice_for_variance.append(np.mean(dice_all))
    return dice_for_variance


@click.command()
@click.option("--hidden-channels", "-H", default=18, type=int)
@click.option(
    "--folds",
    "-f",
    help="Number of folds for k-fold cross validation",
    default=5,
    type=int,
)
@click.option("--id", "-i", help="nnUNet dataset ID", type=int, default=11)
def main(hidden_channels, folds, id):

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

    dice_for_variance_NCA = eval_segmentation_KID_NCA_noise(
        hidden_channels=hidden_channels,
        folds=folds,
        dataset_id=id,
    )

    result_baseline, lesion_size_vs_dice_baseline, dice_for_variance_baseline = (
        eval_segmentation_KID_baselines(folds, id)
    )
    result_nca, lesion_size_vs_dice_nca = eval_segmentation_KID_NCA(
        hidden_channels=hidden_channels,
        folds=folds,
        dataset_id=id,
    )

    dice_for_variance_baseline["NCA"] = dice_for_variance_NCA

    result_baseline.append(result_nca)
    df = pd.DataFrame(result_baseline)
    print(df.round(3))

    lesion_size_vs_dice_baseline["NCA"] = lesion_size_vs_dice_nca

    colors = {**baseline_colors, "NCA": "red"}

    for model_name, dice_for_variance in dice_for_variance_baseline.items():
        linewidth = 1
        linestyle = "dashed"
        if model_name == "NCA":
            linewidth = 3
            linestyle = "solid"
        plt.plot(
            variances,
            dice_for_variance,
            marker="+",
            label=model_name,
            color=colors.get(model_name),
            linestyle=linestyle,
            linewidth=linewidth,
        )
    plt.legend()
    plt.xlabel("GAUSSIAN NOISE VARIANCE", weight="bold", font="Calibri")
    plt.ylabel("ACCURACY [DICE]", weight="bold", font="Calibri")
    plt.show()

    for model_name, lesion_size_vs_dice in lesion_size_vs_dice_baseline.items():
        L = tuple(zip(*sorted(zip(*lesion_size_vs_dice), key=lambda x: x[0])))
        x, y = L
        linewidth = 1
        linestyle = "dashed"
        if model_name == "NCA":
            linewidth = 3
            linestyle = "solid"
        plt.plot(
            x,
            y,
            label=model_name,
            linestyle=linestyle,
            marker="+",
            linewidth=linewidth,
            color=colors.get(model_name),
        )
    plt.xlabel("LESION SIZE [PX]", weight="bold", font="Calibri")
    plt.ylabel("ACCURACY [DICE]", weight="bold", font="Calibri")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    sns.despine()
    plt.show()


if __name__ == "__main__":
    main()
