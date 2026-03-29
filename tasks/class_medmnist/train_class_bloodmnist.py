#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import click
import numpy as np
import torch  # type: ignore[import-untyped]
from medmnist import INFO, BloodMNIST  # type: ignore[import-untyped]
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)


from ncalab import (  # noqa: E402
    fix_random_seed,
    BasicNCATrainer,
    ClassificationNCAModel,
    get_compute_device,
    print_NCALab_banner,
)

TASK_PATH = Path(__file__).parent.resolve()
WEIGHTS_PATH = TASK_PATH / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)

gradient_clipping = False
pad_noise = False
alive_mask = False
use_temporal_encoding = False
fire_rate = 1.0
default_hidden_channels = 20


T_train = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float, scale=True),
        v2.ConvertImageDtype(dtype=torch.float32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize((0.5,), (0.225,)),
    ]
)
T_val = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float, scale=True),
        v2.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize((0.5,), (0.225,)),
    ]
)


def train_class_bloodmnist(
    batch_size: int,
    hidden_channels: int,
    gpu: bool,
    gpu_index: int,
    max_epochs: int,
    dry: bool,
):
    fix_random_seed()
    print_NCALab_banner()

    comment = "BloodMNIST"
    comment += f"_hidden_{hidden_channels}"
    comment += f"_gc_{gradient_clipping}"
    comment += f"_noise_{pad_noise}"
    comment += f"_AM_{alive_mask}"
    comment += f"_TE_{use_temporal_encoding}"

    writer = SummaryWriter(comment=comment) if not dry else None

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    dataset_train = BloodMNIST(split="train", download=True, transform=T_train)
    y_train = dataset_train.labels.squeeze()
    class_sample_count_train = np.array(
        [
            len(np.where(dataset_train.labels == t)[0])
            for t in np.sort(np.unique(y_train))
        ]
    )
    weight = 1.0 / class_sample_count_train
    sample_weight_train = [weight[int(t)].astype(float) for t in y_train]
    sampler_train = torch.utils.data.WeightedRandomSampler(
        sample_weight_train,
        num_samples=len(dataset_train),
        replacement=True,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=batch_size
    )

    dataset_val = BloodMNIST(split="val", download=True, transform=T_val)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=True, batch_size=8
    )

    dataset_test = BloodMNIST(split="test", download=True, transform=T_val)
    loader_test = torch.utils.data.DataLoader(dataset_test)

    num_classes = len(INFO["bloodmnist"]["label"])
    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=num_classes,
        use_alive_mask=alive_mask,
        fire_rate=fire_rate,
        pad_noise=pad_noise,
        use_temporal_encoding=use_temporal_encoding,
        class_names=list(INFO["bloodmnist"]["label"].values()),
        training_timesteps=32,
        inference_timesteps=32,
        use_classifier=True,
        lambda_hidden=1e-2,
    )
    trainer = BasicNCATrainer(
        nca,
        WEIGHTS_PATH / "classification_bloodmnist" if not dry else None,
        batch_repeat=2,
        max_epochs=max_epochs,
        gradient_clipping=gradient_clipping,
    )
    trainer.train(
        loader_train,
        loader_val,
        loader_test,
        summary_writer=writer,
    )
    if writer is not None:
        writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=default_hidden_channels, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
@click.option("--max-epochs", "-E", type=int, default=50)
@click.option("--dry", "-D", is_flag=True)
def main(
    batch_size, hidden_channels, gpu: bool, gpu_index: int, max_epochs: int, dry: bool
):
    train_class_bloodmnist(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
        max_epochs=max_epochs,
        dry=dry,
    )


if __name__ == "__main__":
    main()
