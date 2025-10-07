#!/usr/bin/env python3
import os
import sys

import click
import torch  # type: ignore[import-untyped]
import torchmetrics
import torchmetrics.classification
import torchvision  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]
from tqdm import tqdm
from train_class_cifar10 import (
    TASK_PATH,
    WEIGHTS_PATH,
    alive_mask,
    default_hidden_channels,
    fire_rate,
    pad_noise,
    use_temporal_encoding,
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)


from ncalab import (  # noqa: E402
    ClassificationNCAModel,
    fix_random_seed,
    get_compute_device,
)

T = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float, scale=True),
        v2.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


def eval_class_cifar10(
    hidden_channels: int,
    gpu,
    gpu_index,
):
    fix_random_seed()
    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    testset = torchvision.datasets.CIFAR10(
        root=TASK_PATH / "data", train=False, download=True, transform=T
    )
    loader_test = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )

    class_names = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    num_classes = len(class_names)
    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=num_classes,
        use_alive_mask=alive_mask,
        fire_rate=fire_rate,
        pad_noise=pad_noise,
        use_temporal_encoding=use_temporal_encoding,
        class_names=class_names,
    )
    nca.load_state_dict(
        torch.load(
            WEIGHTS_PATH / "classification_cifar10" / "best_model.pth",
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
        steps = 42

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
    eval_class_cifar10(
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
