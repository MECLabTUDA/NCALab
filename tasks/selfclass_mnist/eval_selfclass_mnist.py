import os
import sys
from pathlib import Path

import click
import numpy as np
import torch
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.datasets import MNIST  # type: ignore[import-untyped]

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import (  # noqa: E402
    SelfClassificationNCAModel,
    VisualBinaryImageClassification,
    fix_random_seed,
    get_compute_device,
    release_random_seed,
)

TASK_PATH = Path(__file__).parent.resolve()
FIGURES_PATH = TASK_PATH / "figures"
FIGURES_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH = TASK_PATH / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)


def print_MNIST_digit(mask: np.ndarray, prediction: np.ndarray, downscale: int = 2):
    assert downscale >= 1
    assert downscale <= 28

    BG = {
        0: "yellow",
        1: "magenta",
        2: "red",
        3: "green",
        4: "yellow",
        5: "cyan",
        6: "green",
        7: "white",
        8: "magenta",
        9: "green",
    }
    FG = {
        0: "white",
        6: "red",
        7: "black",
    }

    for i in range(len(mask)):
        for y in range(28 // downscale):
            for x in range(28 // downscale):
                if mask[i, 0, y * downscale, x * downscale] == 0:
                    click.secho("  ", nl=False, fg="black", bg="black")
                    continue
                n = prediction[i, y * downscale, x * downscale]
                click.secho(
                    f" {n}", nl=False, fg=FG.get(n, "black"), bg=BG.get(n, "white")
                )
            click.secho()
        click.secho("-" * 28)


def eval_selfclass_mnist(
    hidden_channels: int, gpu: bool, gpu_index: int, num_digits: int = 8
):
    fix_random_seed()
    mnist_test = MNIST(
        "mnist",
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    loader_test = torch.utils.data.DataLoader(
        mnist_test, shuffle=True, batch_size=num_digits
    )
    # Randomize seed again. We only rely on the fixed random seed to ensure separation of train/val/test split.
    release_random_seed()


    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    image, label = next(iter(loader_test))
    x = image.to(device)

    nca = SelfClassificationNCAModel(
        device,
        num_hidden_channels=hidden_channels,
        fire_rate=0.8,
        num_classes=10,
        training_timesteps=(40, 60),
        inference_timesteps=50,
    )
    nca.load_state_dict(
        torch.load(
            WEIGHTS_PATH / "selfclass_mnist" / "best_model.pth", weights_only=True
        )
    )
    nca.eval()

    prediction = nca.predict(x)

    vis = VisualBinaryImageClassification()
    fig = vis.show(
        nca, prediction.image_channels_np, prediction, label.detach().cpu().numpy()
    )
    fig.savefig(FIGURES_PATH / "selfclass_mnist_predictions.png", dpi=600)

    class_channels = prediction.output_channels_np
    out_hwc = np.argmax(class_channels, axis=1)
    mask = prediction.image_channels_np > 0.1
    print_MNIST_digit(mask, out_hwc)

    click.secho()
    click.secho(
        f"You should see {num_digits} random downscaled (2x) pictures of\n"
        "handwritten digits (0-9) in the lines above.\n"
        "If they are correctly classified by the NCA, the numbers\n"
        "inside the blocks correspond to the written digit.\n"
    )


@click.command()
@click.option("--hidden-channels", "-H", default=12, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
def main(hidden_channels, gpu, gpu_index):
    eval_selfclass_mnist(
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main()
