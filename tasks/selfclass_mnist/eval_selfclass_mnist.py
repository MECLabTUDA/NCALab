import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import (
    ClassificationNCAModel,
    get_compute_device,
    pad_input,
    WEIGHTS_PATH
)

import click

from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

import torch

from torchvision.datasets import MNIST  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]


def print_MNIST_digit(image, prediction):
    BG = {
        0: "black",
        1: "red",
        2: "cyan",
        3: "green",
        4: "magenta",
        5: "yellow",
        6: "green",
        7: "white",
        8: "blue",
        9: "red"
    }
    FG = {
        0: "white",
        6: "red",
        7: "black",
    }
    for y in range(14):
        for x in range(14):
            if image[y * 2, x * 2] < 0.3:
                click.secho("  ", nl=False, fg="black", bg="black")
                continue
            n = prediction[y * 2, x * 2].detach().cpu()
            n = int(torch.argmax(n))
            click.secho(f" {n}", nl=False, fg=FG.get(n, "black"), bg=BG.get(n, "white"))
        click.secho()


def eval_selfclass_mnist(
    hidden_channels: int, gpu: bool, gpu_index: int
):
    mnist_test = MNIST(
        "mnist",
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    loader_test = torch.utils.data.DataLoader(
        mnist_test, shuffle=True, batch_size=1
    )

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    nca = ClassificationNCAModel(
        device,
        num_image_channels=1,
        num_hidden_channels=hidden_channels,
        num_classes=10,
        pixel_wise_loss=True,
    )
    nca.load_state_dict(torch.load(WEIGHTS_PATH / "selfclass_mnist.pth", weights_only=True))
    nca.eval()

    i = 1
    for image, _ in loader_test:
        if i == 0:
            break
        x = image.clone()
        x = pad_input(x, nca, noise=False)
        x = x.permute(0, 2, 3, 1).to(device)

        prediction = nca(x, steps=50)[0]
        prediction = prediction[..., nca.num_image_channels + nca.num_hidden_channels :]
        print_MNIST_digit(image[0, 0], prediction)

        if i != 1:
            click.secho("-" * 28 * 2)
        i -= 1


@click.command()
@click.option("--hidden-channels", "-H", default=9, type=int)
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
