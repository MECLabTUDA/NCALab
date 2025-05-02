#!/usr/bin/env python3
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

import copy

import click
from tqdm import tqdm


import torch
from torch.utils.tensorboard import SummaryWriter  # for type hint

import albumentations as A  # type: ignore[import-untyped]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-untyped]

import segmentation_models_pytorch as smp

from ncalab import DiceBCELoss, SplitDefinition, WEIGHTS_PATH, get_compute_device
from config import (
    KID_DATASET_PATH,
    KID_SEGMENTATION_MODEL_NAME,
    KID_DATASET_PATH_NNUNET,
)
from kid2dataset import KIDDataset
from baselines import *


def validate_model(device, model, loader, criterion):
    model.eval()
    TP = []
    FP = []
    FN = []
    TN = []
    val_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            tp, fp, fn, tn = smp.metrics.get_stats(
                outputs, labels[:, None, :, :].long(), mode="binary", threshold=0.5
            )
            TP.append(tp[:, 0])
            FP.append(fp[:, 0])
            FN.append(fn[:, 0])
            TN.append(tn[:, 0])
            val_loss += criterion(outputs, labels)
    f1_score = smp.metrics.f1_score(
        torch.cat(TP), torch.cat(FP), torch.cat(FN), torch.cat(TN), reduction="micro"
    )
    iou_score = smp.metrics.iou_score(
        torch.cat(TP), torch.cat(FP), torch.cat(FN), torch.cat(TN), reduction="micro"
    )
    return {"f1": f1_score.item(), "iou": iou_score.item()}


class Trainer:
    def train(
        self,
        device,
        model,
        model_name,
        train_loader,
        val_loader,
        fold: int,
        criterion,
        optimizer,
        num_epochs=100,
        patience=20,
    ):
        model.to(device)
        best_accuracy = 0.0
        best_model_weights = None
        patience_countdown = patience
        writer = SummaryWriter()

        print("Saving weights as", WEIGHTS_PATH / f"unet_{model_name}_fold{fold:02d}.pth")

        for epoch in tqdm(range(1, num_epochs + 1)):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            val_f1 = validate_model(device, model, val_loader, criterion)["f1"]

            print(
                f"Epoch {epoch}/{num_epochs}, "
                f"Training Loss: {epoch_loss:.4f}, "
                f"Validation Accuracy: {val_f1:.4f}\n"
            )

            if val_f1 > best_accuracy:
                best_accuracy = val_f1
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(
                    best_model_weights,
                    WEIGHTS_PATH / f"unet_{model_name}_fold{fold:02d}.pth",
                )
                patience_countdown = patience
            else:
                patience_countdown -= 1
                if patience_countdown == 0:
                    print("--> Early Stopping")
                    break
        writer.close()


@click.command()
@click.option(
    "--folds",
    "-f",
    help="Number of folds for k-fold cross validation",
    default=5,
    type=int,
)
@click.option("--dataset-id", "-i", help="nnUNet dataset ID", type=int, default=11)
def train_baselines(folds, dataset_id):
    device = get_compute_device("cuda:0")

    T = A.Compose(
        [
            A.CenterCrop(320, 320),
            A.RandomRotate90(),
            ToTensorV2(),
        ]
    )

    zoo = make_model_zoo()

    split = SplitDefinition.read(
        KID_DATASET_PATH_NNUNET
        / "nnUNet_preprocessed"
        / f"Dataset{dataset_id:03d}_KID2vascular"
        / "splits_final.json"
    )

    for model_name, model in zoo.items():
        print(f"Training model {model_name}")
        for fold in range(folds):
            print(f"Fold {fold}")
            dataloaders = split[fold].dataloaders(
                KIDDataset,
                KID_DATASET_PATH_NNUNET
                / "nnUNet_raw"
                / f"Dataset{dataset_id:03d}_KID2vascular",
                T,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            trainer = Trainer()
            trainer.train(
                device,
                model,
                model_name,
                dataloaders["train"],
                dataloaders["val"],
                fold,
                criterion=DiceBCELoss(),
                optimizer=optimizer,
            )


if __name__ == "__main__":
    train_baselines()
