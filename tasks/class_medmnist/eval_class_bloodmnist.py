#!/usr/bin/env python3
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import ClassificationNCAModel, get_compute_device

import click

from medmnist import INFO, BloodMNIST  # type: ignore[import-untyped]

import torch  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]

import torchmetrics
import torchmetrics.classification

from tqdm import tqdm

from train_class_bloodmnist import (
    pad_noise,
    alive_mask,
    use_temporal_encoding,
    fire_rate,
    WEIGHTS_PATH,
    default_hidden_channels,
)

T = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float, scale=True),
        v2.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize((0.5,), (0.225,)),
    ]
)


def eval_selfclass_bloodmnist(
    hidden_channels: int,
    gpu,
    gpu_index,
):
    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    dataset_test = BloodMNIST(split="test", download=True, transform=T)
    loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False, batch_size=32
    )

    num_classes = len(INFO["bloodmnist"]["label"])
    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        hidden_size=128,
        num_classes=num_classes,
        use_alive_mask=alive_mask,
        fire_rate=fire_rate,
        use_temporal_encoding=use_temporal_encoding,
        pad_noise=pad_noise,
    )
    nca.load_state_dict(
        torch.load(
            WEIGHTS_PATH / "classification_bloodmnist" / "best_model.pth",
            weights_only=True,
        )
    )

    params = nca.num_trainable_parameters()
    print(f"Trainable parameters: {params}")
    print(f"That is {4 * params / 1000} kB")

    nca = nca.to(device)
    nca.eval()

    macro_acc = torchmetrics.classification.MulticlassAccuracy(
        average="macro", num_classes=num_classes
    ).to(device)
    micro_acc = torchmetrics.classification.MulticlassAccuracy(
        average="micro", num_classes=num_classes
    ).to(device)
    macro_auc = torchmetrics.classification.MulticlassAUROC(
        average="macro",
        num_classes=num_classes,
    ).to(device)
    micro_f1 = torchmetrics.classification.MulticlassF1Score(
        average="micro", num_classes=num_classes
    ).to(device)

    for sample in tqdm(iter(loader_test)):
        x, y = sample
        x = x.float().to(device)
        steps = 32

        y_prob = nca.classify(x, steps, reduce=False)
        y = y.squeeze().to(device)

        macro_acc.update(y_prob, y)
        micro_acc.update(y_prob, y)
        macro_auc.update(y_prob, y)
        micro_f1.update(y_prob, y)
    macro_acc_ = macro_acc.compute().item()
    micro_acc_ = micro_acc.compute().item()
    macro_auc_ = macro_auc.compute().item()
    micro_f1_ = micro_f1.compute().item()

    print()
    print(
        f"ACC macro: {macro_acc_:.3f}  ACC micro: {micro_acc_:.3f}  AUC macro: {macro_auc_:.3f}  F1 micro: {micro_f1_:.3f}"
    )


@click.command()
@click.option("--hidden-channels", "-H", default=default_hidden_channels, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def main(hidden_channels, gpu: bool, gpu_index: int):
    eval_selfclass_bloodmnist(
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
