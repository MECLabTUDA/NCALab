import os
import sys
from pathlib import Path

import click
import torch
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.datasets import MNIST  # type: ignore[import-untyped]

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from ncalab import (  # noqa: E402
    BasicNCATrainer,
    ClassificationNCAModel,
    fix_random_seed,
    get_compute_device,
    print_mascot,
    print_NCALab_banner,
)

TASK_PATH = Path(__file__).parent.resolve()
WEIGHTS_PATH = TASK_PATH / "weights"
WEIGHTS_PATH.mkdir(exist_ok=True)


def train_selfclass_mnist(
    batch_size: int,
    hidden_channels: int,
    gpu: bool,
    gpu_index: int,
    max_epochs: int,
    dry: bool,
):
    print_NCALab_banner()
    print()
    print_mascot(
        "While this model is training, you may like to read the\n"
        " original article about self-classifying MNIST digits:\n"
        "\n"
        "https://distill.pub/2020/selforg/mnist/\n"
        "(Ctrl + click to open in browser)"
    )
    fix_random_seed()

    writer = SummaryWriter(comment="Selfclassifying MNIST") if not dry else None

    mnist_train = MNIST(
        "mnist",
        download=True,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    # Split MNIST dataset into training and validation
    train_indices, _, val_indices, _ = train_test_split(
        range(len(mnist_train)),
        mnist_train.targets,
        stratify=mnist_train.targets,
        test_size=int(len(mnist_train) * 0.2),  # TODO configure via CLI
    )

    train_split = Subset(mnist_train, train_indices)
    val_split = Subset(mnist_train, val_indices)

    loader_train = torch.utils.data.DataLoader(
        train_split, shuffle=True, batch_size=batch_size
    )
    loader_val = torch.utils.data.DataLoader(val_split, shuffle=True, batch_size=32)

    device = get_compute_device(f"cuda:{gpu_index}" if gpu else "cpu")

    nca = ClassificationNCAModel(
        device,
        num_image_channels=1,
        num_hidden_channels=hidden_channels,
        num_classes=10,
        pixel_wise_loss=True,
        use_classifier=False,
        training_timesteps=(40, 60),
        inference_timesteps=50,
    )

    trainer = BasicNCATrainer(
        nca,
        WEIGHTS_PATH / "selfclass_mnist" if not dry else None,
        max_epochs=max_epochs,
    )
    trainer.train(
        loader_train,
        loader_val,
        summary_writer=writer,
    )
    if writer is not None:
        writer.close()


@click.command()
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--hidden-channels", "-H", default=9, type=int)
@click.option(
    "--gpu/--no-gpu", is_flag=True, default=True, help="Try using the GPU if available."
)
@click.option(
    "--gpu-index", type=int, default=0, help="Index of GPU to use, if --gpu in use."
)
@click.option("--max-epochs", "-E", type=int, default=5)
@click.option("--dry", "-D", is_flag=True)
def main(batch_size, hidden_channels, gpu, gpu_index, max_epochs: int, dry: bool):
    train_selfclass_mnist(
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        gpu=gpu,
        gpu_index=gpu_index,
        max_epochs=max_epochs,
        dry=dry,
    )


if __name__ == "__main__":
    main()
