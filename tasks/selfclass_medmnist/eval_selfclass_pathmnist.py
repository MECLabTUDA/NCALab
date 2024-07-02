#!/usr/bin/env python3
import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from nca import ClassificationNCAModel, WEIGHTS_PATH, get_compute_device, pad_input

import click

import torch

from medmnist import PathMNIST

from torchvision import transforms
from torchvision.transforms import v2

from torch.utils.tensorboard import SummaryWriter

import numpy as np

from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score

from tqdm import tqdm


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
                v2.ToImageTensor(),
                v2.ConvertImageDtype(dtype=torch.float32),
                # transforms.Normalize(
                #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        ),
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False, batch_size=32
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
    nca.load_state_dict(torch.load(WEIGHTS_PATH / "selfclass_pathmnist.best.pth"))

    model_parameters = filter(lambda p: p.requires_grad, nca.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Trainable parameters: {params}")
    print(f"That is {4 * params / 1000} kB")

    nca = nca.to(device)
    nca.eval()

    macro_acc = MulticlassAccuracy(average="macro", num_classes=9)
    micro_acc = MulticlassAccuracy(average="micro", num_classes=9)
    macro_auc = MulticlassAUROC(
        average="macro",
        num_classes=9,
    )
    micro_f1 = MulticlassF1Score(average="micro", num_classes=9)
    for sample in tqdm(iter(loader_test)):
        x, y = sample
        x = pad_input(x, nca, noise=True)
        x = x.float().transpose(1, 3).to(device)
        steps = 70#np.random.randint(64, 96)
        y_prob = torch.stack(
            [nca.classify(x, steps, reduce=False) for _ in range(1)]
        ).mean(dim=0)

        y = y.squeeze().to(device)
        macro_acc.update(y_prob, y)
        micro_acc.update(y_prob, y)
        macro_auc.update(y_prob, y)
        micro_f1.update(y_prob, y)
    macro_acc_ = macro_acc.compute().item()
    micro_acc_ = micro_acc.compute().item()
    macro_auc_ = macro_auc.compute().item()
    micro_f1_ = micro_f1.compute().item()
    print(
        f"ACC macro: {macro_acc_:.3f}  ACC micro: {micro_acc_:.3f}  AUC macro: {macro_auc_:.3f}  F1 micro: {micro_f1_:.3f}"
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
