import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import ClassificationNCAModel, get_compute_device, pad_input, WEIGHTS_PATH

import click

import torch

from torchvision.datasets import MNIST  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]


def print_MNIST_digit(image, prediction, downscale: int = 2):
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

    for y in range(28 // downscale):
        for x in range(28 // downscale):
            if image[y * downscale, x * downscale] < 0.3:
                click.secho("  ", nl=False, fg="black", bg="black")
                continue
            n = prediction[y * downscale, x * downscale].detach().cpu()
            n = int(torch.argmax(n))
            click.secho(f" {n}", nl=False, fg=FG.get(n, "black"), bg=BG.get(n, "white"))
        click.secho()


def eval_selfclass_mnist(
    hidden_channels: int, gpu: bool, gpu_index: int, num_digits: int = 3
):
    mnist_test = MNIST(
        "mnist",
        download=True,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    loader_test = torch.utils.data.DataLoader(mnist_test, shuffle=True, batch_size=1)

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    nca = ClassificationNCAModel(
        device,
        num_image_channels=1,
        num_hidden_channels=hidden_channels,
        num_classes=10,
        pixel_wise_loss=True,
    )
    nca.load_state_dict(
        torch.load(WEIGHTS_PATH / "selfclass_mnist.pth", weights_only=True)
    )
    nca.eval()

    i = num_digits
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
            click.secho("-" * 28)
        i -= 1

    click.secho()
    click.secho(
        f"You should see {num_digits} random downscaled (2x) pictures of\n"
        "handwritten digits (0-9) in the lines above.\n"
        "If they are correctly classified by the NCA, the numbers\n"
        "inside the blocks correspond to the written digit.\n"
    )


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
