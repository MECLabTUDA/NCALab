#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca import (
    ClassificationNCAModel,
    train_basic_nca,
    WEIGHTS_PATH,
    show_batch_classification,
    get_compute_device,
)

import click

import torch

from medmnist import PathMNIST

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC


def eval_selfclass_pathmnist(
    hidden_channels: int,
    gpu,
    gpu_index,
):
    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    dataset_test = PathMNIST(
        split="test",
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # automatic divide by 255
            ]
        ),
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False, batch_size=256
    )

    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        hidden_size=128,
        num_classes=9,
        use_alive_mask=False,
        fire_rate=0.5,
    )
    nca.load_state_dict(torch.load(WEIGHTS_PATH / "selfclass_pathmnist.pth"))
    nca = nca.to(device)

    macro_acc = MulticlassAccuracy(average="macro", num_classes=9)
    micro_acc = MulticlassAccuracy(average="micro", num_classes=9)
    macro_auc = MulticlassAUROC(
        average="macro",
        num_classes=9,
    )
    #micro_auc = MulticlassAUROC(
    #    average="micro",
    #    num_classes=9,
    #)
    for sample in iter(loader_test):
        x, y = sample
        if x.shape[1] < nca.num_channels:
            x = np.pad(
                x,
                [
                    (0, 0),  # batch
                    (0, nca.num_channels - x.shape[1]),  # channels
                    (0, 0),  # width
                    (0, 0),  # height
                ],
                mode="constant",
            )
            x = torch.from_numpy(x.astype(np.float32))
        x = x.float().transpose(1, 3)
        y_pred = nca.classify(x.to(device), 100, reduce=True)
        y_prob = nca.classify(x.to(device), 100, reduce=False)

        macro_acc.update(y_pred, y.squeeze().to(device))
        micro_acc.update(y_prob, y.squeeze().to(device))
        macro_auc.update(y_prob, y.squeeze().to(device))
        #micro_auc.update(y_prob, y.squeeze().to(device))
    macro_acc_ = macro_acc.compute().item()
    micro_acc_ = micro_acc.compute().item()
    macro_auc_ = macro_auc.compute().item()
    #micro_auc_ = micro_auc.compute().item()
    print(
        f"ACC macro: {macro_acc_:.3f}  ACC micro: {micro_acc_:.3f}  AUC macro: {macro_auc_:.3f}"
    )


@click.command()
@click.option("--hidden-channels", "-H", default=16, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def main(hidden_channels, gpu: bool, gpu_index: int):
    eval_selfclass_pathmnist(
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
