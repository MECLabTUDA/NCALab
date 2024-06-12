from pathlib import PosixPath  # for type hint
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


def train_basic_nca(
    nca: BasicNCAModel,
    model_path: str | PosixPath,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader | None = None,
    max_iterations: int = 50000,
    gradient_clipping: bool = True,
    steps_range: tuple = (64, 96),
    steps_validation: int = 80,
    save_every: int = 100,
    lr: float = 2e-3,
    lr_gamma: float = 0.9999,
    adam_betas=(0.5, 0.5),
    summary_writer: SummaryWriter = None,
    plot_function: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, BasicNCAModel], Figure] | None
    ) = None,
    batch_repeat: int = 2,
):
    """Execute basic NCA training loop with a single function call.

    Args:
        nca (BasicNCAModel): Model instance to train. Should be based on BasicNCAModel.
        model_path (str | PosixPath): File path to store model weights during training.
        dataloader_train (DataLoader): Training DataLoader
        dataloader_val (DataLoader): Validation DataLoader
        max_iterations (int, optional): _description_. Defaults to 50000.
        gradient_clipping (bool, optional): _description_. Defaults to True.
        steps_range (tuple, optional): _description_. Defaults to (64, 96).
        steps_validation (int, optional): Forward passes during validation. Defaults to 80.
        save_every (int, optional): _description_. Defaults to 100.
        lr (float, optional): Start learning rate. Defaults to 2e-3.
        lr_gamma (float, optional): _description_. Defaults to 0.9999.
        adam_betas (tuple, optional): _description_. Defaults to (0.5, 0.5).
        summary_writer (SummaryWriter, optional): Tensorboard SummaryWriter. Defaults to None.
        plot_function (Callable[ [np.ndarray, np.ndarray, np.ndarray, BasicNCAModel], Figure ], optional): _description_. Defaults to None.
        batch_multiplier (int, optional): How often a batch should be repeated, minimum is 1. Batch duplication can stabelize the training. Defaults to 2.
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
        optimizer.zero_grad()
        x_pred = x.clone().to(nca.device)
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
        sample = next(iter(dataloader_train))
        x, y = sample

        # Typically, our dataloader supplies a binary, grayscale, RGB or RGBA image.
        # But out NCA operates on multiple hidden channels and output channels, so we
        # need to pad the input image with zeros.
        if x.shape[1] < nca.num_channels:
            x = np.pad(
                x,
                [
                    (0, 0),  # batch
                    (0, nca.num_channels - x.shape[1]),  # channels
                    (0, 0),  # width
                    (0, 0),  # height
                ],
                mode="constant",
            )
            x = torch.from_numpy(x.astype(np.float32))
        x = x.float().transpose(1, 3)

        # batch duplication
        x = torch.cat(batch_repeat * [x])
        y = torch.cat(batch_repeat * [y])

        steps = np.random.randint(*steps_range)
        x_pred = train_iteration(x, y, steps, optimizer, scheduler, iteration)

        if iteration % save_every == 0:
            torch.save(nca.state_dict(), model_path)
            if plot_function:
                figure = plot_function(
                    x.detach().cpu().numpy(),
                    x_pred.detach().cpu().numpy(),
                    y.detach().cpu().numpy(),
                    nca,
                )
                summary_writer.add_figure("Current Batch", figure, iteration)

        if dataloader_val:
            sample = next(iter(dataloader_val))
            x, y = sample
            if x.shape[1] < nca.num_channels:
                x = np.pad(
                    x,
                    [
                        (0, 0),  # batch
                        (0, nca.num_channels - x.shape[1]),  # channels
                        (0, 0),  # width
                        (0, 0),  # height
                    ],
                    mode="constant",
                )
                x = torch.from_numpy(x.astype(np.float32))
            x = x.float().transpose(1, 3)
            nca.validate(x, y, steps_validation, iteration, summary_writer)
