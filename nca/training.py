import io
from pathlib import PosixPath  # for type hint
from typing import Callable, Iterable, Dict

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader  # for type hint

import matplotlib.pyplot as plt
from matplotlib.figure import Figure  # for type hint
import PIL.Image
from torchvision.transforms import ToTensor

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
    save_every: int = 100,
    lr: float = 2e-3,
    lr_gamma: float = 0.9999,
    adam_betas=(0.5, 0.5),
    summary_writer=None,
    plot_function: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, BasicNCAModel], Figure] | None
    ) = None,
):
    """_summary_

    Args:
        nca (BasicNCAModel): _description_
        model_path (str | PosixPath): _description_
        dataloader_train (DataLoader): Training DataLoader
        dataloader_val (DataLoader): Validation DataLoader
        max_iterations (int, optional): _description_. Defaults to 50000.
        gradient_clipping (bool, optional): _description_. Defaults to True.
        steps_range (tuple, optional): _description_. Defaults to (64, 96).
        save_every (int, optional): _description_. Defaults to 100.
        lr (float, optional): _description_. Defaults to 2e-3.
        lr_gamma (float, optional): _description_. Defaults to 0.9999.
        adam_betas (tuple, optional): _description_. Defaults to (0.5, 0.5).
        summary_writer (_type_, optional): _description_. Defaults to None.
        pad_x (bool, optional): _description_. Defaults to False.
        plot_function (Callable[ [np.ndarray, np.ndarray, np.ndarray, BasicNCAModel], Figure ], optional): _description_. Defaults to None.
    """

    if not plot_function:
        if nca.plot_function:
            plot_function = nca.plot_function

    optimizer = optim.Adam(nca.parameters(), lr=lr, betas=adam_betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

    def train_iteration(x, target, steps, optimizer, scheduler, batch_iteration):
        optimizer.zero_grad()
        x_pred = x.clone().to(nca.device)
        x_pred = nca(x_pred, steps=steps)

        loss = nca.loss(x_pred, target.to(nca.device))
        loss.backward()

        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(nca.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if summary_writer:
            summary_writer.add_scalar("Loss/train", loss, batch_iteration)
        return x_pred, loss

    def val_iteration(x, target, steps, batch_iteration):
        with torch.no_grad():
            x_pred = x.clone().to(nca.device)
            x_pred = nca(x_pred, steps=steps)
            loss = nca.loss(x_pred, target.to(nca.device))
            if summary_writer:
                summary_writer.add_scalar("Loss/val", loss, batch_iteration)

    for iteration in tqdm(range(max_iterations)):
        sample = next(iter(dataloader_train))
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

        steps = np.random.randint(*steps_range)
        x_pred, loss = train_iteration(x, y, steps, optimizer, scheduler, iteration)

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
            val_iteration(x, y, 80, iteration)