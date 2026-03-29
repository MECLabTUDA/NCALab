#!/usr/bin/env python3
import os
import pprint
import sys

import click
import torch  # type: ignore[import-untyped]
import torchvision  # type: ignore[import-untyped]
from train_class_cifar10 import (
    TASK_PATH,
    WEIGHTS_PATH,
    alive_mask,
    default_hidden_channels,
    fire_rate,
    pad_noise,
    use_temporal_encoding,
    T_val,
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)


from ncalab import (  # noqa: E402
    Animator,
    ClassificationNCAModel,
    CascadeNCA,
    fix_random_seed,
    get_compute_device,
)

FIGURE_PATH = TASK_PATH / "figures"
FIGURE_PATH.mkdir(exist_ok=True)


def eval_class_cifar10(
    hidden_channels: int,
    gpu,
    gpu_index,
):
    fix_random_seed()
    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    testset = torchvision.datasets.CIFAR10(
        root=TASK_PATH / "data", train=False, download=True, transform=T_val
    )
    loader_test = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=False, num_workers=4
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
        filter_padding="circular",
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_classes=num_classes,
        use_alive_mask=alive_mask,
        fire_rate=fire_rate,
        pad_noise=pad_noise,
        use_temporal_encoding=use_temporal_encoding,
        class_names=class_names,
        training_timesteps=48,
        inference_timesteps=48,
        use_classifier=True,
    )
    cascade = CascadeNCA(nca, [4, 2, 1], [16, 8, 8])
    cascade.load_state_dict(
        torch.load(
            WEIGHTS_PATH / "classification_cifar10" / "best_model.pth",
            weights_only=True,
        )
    )

    params = cascade.num_trainable_parameters()
    print(f"Trainable parameters: {params}")
    print(f"That is {4 * params / 1000} kB")

    metrics, _ = cascade.validate(loader_test)

    seed = next(iter(loader_test))[0].to(device)
    out_path = FIGURE_PATH / "classification_cifar10.gif"
    animator = Animator(nca, seed, interval=100, show_input=True, hidden=True)
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
    eval_class_cifar10(
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
