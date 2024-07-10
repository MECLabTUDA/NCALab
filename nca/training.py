from pathlib import Path, PosixPath  # for type hint
from typing import Callable

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader  # for type hint
from torch.utils.tensorboard import SummaryWriter  # for type hint

from matplotlib.figure import Figure  # for type hint

from tqdm import tqdm

from .models.basicNCA import BasicNCAModel  # for type hint
from .utils import pad_input


def train_basic_nca(
    nca: BasicNCAModel,
    model_path: str | Path | PosixPath,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader | None = None,
    max_iterations: int = 5000,
    gradient_clipping: bool = False,
    steps_range: tuple = (64, 96),
    steps_validation: int = 80,
    save_every: int = 5,
    lr: float = 2e-3,
    lr_gamma: float = 0.9999,
    adam_betas=(0.5, 0.5),
    summary_writer: SummaryWriter | None = None,
    plot_function: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, BasicNCAModel], Figure] | None
    ) = None,
    batch_repeat: int = 2,
    truncate_backprop: bool = False,
    pad_noise: bool = False,
):
    """Execute basic NCA training loop with a single function call.

    Args:
        nca (BasicNCAModel): Model instance to train. Should be based on BasicNCAModel.
        model_path (str | PosixPath): File path to store model weights during training.
        dataloader_train (DataLoader): Training DataLoader
        dataloader_val (DataLoader): Validation DataLoader
        max_iterations (int, optional): Maximum number of batch iterations. Defaults to 50000.
        gradient_clipping (bool, optional): Whether to clip gradients to 1.0 during training. Defaults to True.
        steps_range (tuple, optional): Exclusive range from which to uniformly sample number of steps. Defaults to (64, 96).
        steps_validation (int, optional): Forward passes during validation. Defaults to 80.
        save_every (int, optional): Save every N iterations. Defaults to 100.
        lr (float, optional): Start learning rate. Defaults to 2e-3.
        lr_gamma (float, optional): Learning rate decay over time. Defaults to 0.9999.
        adam_betas (tuple, optional): _description_. Defaults to (0.5, 0.5).
        summary_writer (SummaryWriter, optional): Tensorboard SummaryWriter. Defaults to None.
        plot_function (Callable[ [np.ndarray, np.ndarray, np.ndarray, BasicNCAModel], Figure ], optional): _description_. Defaults to None.
        batch_repeat (int, optional): How often a batch should be repeated, minimum is 1. Batch duplication can stabelize the training. Defaults to 2.
    """

    assert batch_repeat >= 1
    assert lr > 0
    assert steps_range[0] < steps_range[1]
    assert save_every > 0
    assert max_iterations > 0

    # Use default plot function for NCA flavor if none is explicitly given
    if not plot_function:
        if nca.plot_function:
            plot_function = nca.plot_function

    optimizer = optim.Adam(nca.parameters(), lr=lr, betas=adam_betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)
    best_acc = 0
    best_path = Path(model_path).with_suffix(".best.pth")

    def train_iteration(
        x: torch.Tensor,
        y: torch.Tensor,
        steps: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        batch_iteration: int,
    ) -> torch.Tensor:
        """Run a single training iteration.

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            steps (int): Number of inference steps.
            optimizer (torch.optim.Optimizer): _description_
            scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            batch_iteration (int): _description_

        Returns:
            torch.Tensor: Predicted image.
        """
        nca.train()
        optimizer.zero_grad()
        x_pred = x.clone().to(nca.device)
        if truncate_backprop:
            for step in range(steps):
                x_pred = nca(x_pred, steps=1)
                if step < steps - 10:
                    x_pred.detach()
        else:
            x_pred = nca(x_pred, steps=steps)
        loss = nca.loss(x_pred, y.to(nca.device))
        loss.backward()

        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(nca.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if summary_writer:
            summary_writer.add_scalar("Loss/train", loss, batch_iteration)
        return x_pred

    # Main training/validation loop
    for iteration in tqdm(range(max_iterations)):
        # disable tqdm progress bar if dataset has only one sample, e.g. in the growing task
        gen = iter(dataloader_train)
        if len(dataloader_train.dataset) > dataloader_train.batch_size:
            gen = tqdm(gen)
        for i, sample in enumerate(gen):
            x, y = sample

            # Typically, our dataloader supplies a binary, grayscale, RGB or RGBA image.
            # But out NCA operates on multiple hidden channels and output channels, so we
            # need to pad the input image with zeros.
            x = pad_input(x, nca, noise=pad_noise)
            x = x.transpose(1, 3)  # B W H C
            if len(y.shape) == 3:
                y = y.transpose(1, 2)

            # batch duplication, slightly stabelizes the training
            x = torch.cat(batch_repeat * [x])
            y = torch.cat(batch_repeat * [y])

            steps = np.random.randint(*steps_range)
            x_pred = train_iteration(x, y, steps, optimizer, scheduler, iteration)

        if plot_function and (iteration + 1) % save_every == 0:
            figure = plot_function(
                x.detach().cpu().numpy(),
                x_pred.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                nca,
            )
            summary_writer.add_figure("Training Batch", figure, iteration)

        with torch.no_grad():
            nca.eval()
            torch.save(nca.state_dict(), model_path)

            if dataloader_val:
                val_acc = nca.validate(
                    dataloader_val,
                    steps_validation,
                    iteration,
                    summary_writer,
                    pad_noise,
                )
                if val_acc > best_acc:
                    print(f"improved: {best_acc:.5f} --> {val_acc:.5f}")
                    torch.save(nca.state_dict(), best_path)
                    best_acc = val_acc
