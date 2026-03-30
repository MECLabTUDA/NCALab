#!/usr/bin/env python3
import os
import pprint
import sys

import click
import torch  # type: ignore[import-untyped]
from medmnist import INFO, PathMNIST  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]
from train_class_pathmnist import (
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
    Animator,
    ClassificationNCAModel,
    fix_random_seed,
    get_compute_device,
)

FIGURE_PATH = TASK_PATH / "figures"
FIGURE_PATH.mkdir(exist_ok=True)

T = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float, scale=True),
        v2.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize((0.5,), (0.225,)),
    ]
)


def eval_selfclass_pathmnist(
    hidden_channels: int,
    gpu,
    gpu_index,
):
    fix_random_seed()
    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    dataset_test = PathMNIST(split="test", download=True, transform=T)
    loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False, batch_size=8
    )

    num_classes = len(INFO["pathmnist"]["label"])
    nca = ClassificationNCAModel(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=num_classes,
        use_alive_mask=alive_mask,
        fire_rate=fire_rate,
        pad_noise=pad_noise,
        use_temporal_encoding=use_temporal_encoding,
        class_names=list(INFO["pathmnist"]["label"].values()),
        training_timesteps=32,
        inference_timesteps=32,
        use_classifier=True,
    )
    nca.load_state_dict(
        torch.load(
            WEIGHTS_PATH / "classification_pathmnist" / "best_model.pth",
            weights_only=True,
        )
    )
    params = nca.num_trainable_parameters()
    print(f"Trainable parameters: {params}")
    print(f"That is {4 * params / 1000} kB")

    metrics, _ = nca.validate(loader_test)

    seed = next(iter(loader_test))[0].to(device)
    out_path = FIGURE_PATH / "classification_pathmnist.gif"
    animator = Animator(nca, seed, interval=100, hidden=True, show_input=True)
    animator.save(out_path)
    print(f"Animation saved to: {out_path}")

    print()
    pprint.pprint(metrics)


@click.command()
@click.option("--hidden-channels", "-H", default=default_hidden_channels, type=int)
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
