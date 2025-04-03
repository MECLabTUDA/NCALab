from __future__ import annotations
import copy
import logging
from pathlib import Path, PosixPath  # for type hint
from typing import Callable, Optional

import numpy as np

import torch  # type: ignore[import-untyped]
import torch.optim as optim  # type: ignore[import-untyped]
from torch.utils.data import DataLoader  # for type hint
from torch.utils.tensorboard import SummaryWriter  # for type hint

from matplotlib.figure import Figure  # type: ignore[import-untyped]

from tqdm import tqdm  # type: ignore[import-untyped]

from ..models.basicNCA import BasicNCAModel  # for type hint
from ..utils import pad_input

from .earlystopping import EarlyStopping
from .trainingsummary import TrainingSummary


class BasicNCATrainer:
    """
    Trainer class for any model subclassing BasicNCA.
    """

    def __init__(
        self,
        nca: BasicNCAModel,
        model_path: Optional[str | Path | PosixPath] = None,
        gradient_clipping: bool = False,
        steps_range: tuple = (90, 110),
        steps_validation: int = 100,
        lr: float = 16e-4,
        lr_gamma: float = 0.9999,
        adam_betas=(0.9, 0.99),
        batch_repeat: int = 2,
        truncate_backprop: bool = False,
        max_epochs: int = 200,
        p_retain_pool: float = 0.0,
    ):
        """_summary_

        Args:
            nca (BasicNCAModel): _description_
            model_path (str | Path | PosixPath | None, optional): _description_. Defaults to None.
            gradient_clipping (bool, optional): _description_. Defaults to False.
            steps_range (tuple, optional): _description_. Defaults to (90, 110).
            steps_validation (int, optional): _description_. Defaults to 100.
            lr (float, optional): _description_. Defaults to 16e-4.
            lr_gamma (float, optional): _description_. Defaults to 0.9999.
            adam_betas (tuple, optional): _description_. Defaults to (0.9, 0.99).
            batch_repeat (int, optional): _description_. Defaults to 2.
            truncate_backprop (bool, optional): _description_. Defaults to False.
            max_epochs (int, optional): _description_. Defaults to 200.
            p_retain_pool (float, optional): _description_. Defaults to 0.0.
        """
        assert batch_repeat >= 1
        assert lr > 0
        assert steps_range[0] < steps_range[1]
        assert max_epochs > 0
        assert p_retain_pool >= 0.0 and p_retain_pool <= 1.0
        self.nca = nca
        self.model_path = model_path
        self.gradient_clipping = gradient_clipping
        self.steps_range = steps_range
        self.steps_validation = steps_validation
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.adam_betas = adam_betas
        self.batch_repeat = batch_repeat
        self.truncate_backprop = truncate_backprop
        self.max_epochs = max_epochs
        self.p_retain_pool = p_retain_pool

    def info(self) -> str:
        """
        Shows a markdown-formatted info string with training parameters.
        Useful for showing info on tensorboard to keep track of parameter changes.

        Returns:
            str: Markdown-formatted info string.
        """
        s = "BasicNCATrainer Info\n"
        s += "-------------------\n"
        for attribute in (
            "model_path",
            "lr",
            "lr_gamma",
            "gradient_clipping",
            "adam_betas",
            "batch_repeat",
            "truncate_backprop",
            "max_epochs",
            "p_retain_pool",
        ):
            attribute_f = attribute.title().replace("_", " ")
            s += f"**{attribute_f}:** {getattr(self, attribute)}\n"
        return s

    def train(
        self,
        dataloader_train: DataLoader,
        dataloader_val: Optional[DataLoader] = None,
        dataloader_test: Optional[DataLoader] = None,
        save_every: Optional[int] = None,
        summary_writer: Optional[SummaryWriter] = None,
        plot_function: Optional[
            Callable[[np.ndarray, np.ndarray, np.ndarray, BasicNCAModel], Figure]
        ] = None,
        earlystopping: Optional[EarlyStopping] = None,
    ) -> TrainingSummary:
        """Execute basic NCA training loop with a single function call.

        Args:
            dataloader_train (DataLoader): Training DataLoader
            dataloader_val (DataLoader): Validation DataLoader
            save_every (int):
            summary_writer (SummaryWriter, optional): Tensorboard SummaryWriter. Defaults to None.
            plot_function (Callable[ [np.ndarray, np.ndarray, np.ndarray, BasicNCAModel], Figure ], optional): _description_. Defaults to None.
        """
        logging.basicConfig(encoding="utf-8", level=logging.INFO)

        if save_every is None:
            save_every = 100
        assert save_every > 0

        # Use default plot function for NCA flavor if none is explicitly given
        if not plot_function:
            if self.nca.plot_function:
                plot_function = self.nca.plot_function

        if summary_writer is not None:
            summary_writer.add_text("Training Info", self.info())

        optimizer = optim.Adam(self.nca.parameters(), lr=self.lr, betas=self.adam_betas)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.lr_gamma)
        best_acc = 0.0
        best_model = self.nca
        if self.model_path:
            best_path = Path(self.model_path).with_suffix(".best.pth")
        else:
            best_path = None

        def train_iteration(
            x: torch.Tensor,
            y: torch.Tensor,
            steps: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            total_batch_iterations: int,
        ) -> torch.Tensor:
            """Run a single training iteration.

            Args:
                x (torch.Tensor): Input training images.
                y (torch.Tensor): Input training labels.
                steps (int): Number of NCA inference time steps.
                optimizer (torch.optim.Optimizer): Optimizer.
                scheduler (torch.optim.lr_scheduler.LRScheduler): Scheduler.
                total_batch_iterations (int): Total training batch iterations

            Returns:
                torch.Tensor: Predicted image.
            """
            device = self.nca.device
            self.nca.train()
            optimizer.zero_grad()
            x_pred = x.clone().to(self.nca.device)
            if self.truncate_backprop:
                for step in range(steps):
                    x_pred = self.nca(x_pred, steps=1)
                    if step < steps - 10:
                        x_pred.detach()
            else:
                x_pred = self.nca(x_pred, steps=steps)
            losses = self.nca.loss(x_pred, y.to(device))
            losses["total"].backward()

            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.nca.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if summary_writer:
                for key in losses:
                    summary_writer.add_scalar(
                        f"Loss/train_{key}", losses[key], total_batch_iterations
                    )
            return x_pred

        # MAIN LOOP
        total_batch_iterations = 0
        for iteration in tqdm(range(self.max_epochs), desc="Epochs"):
            if earlystopping is not None:
                if earlystopping.done():
                    break

            # disable tqdm progress bar if dataset has only one sample, e.g. in the growing task
            gen = iter(dataloader_train)
            if len(dataloader_train) > 3:
                gen = tqdm(gen, desc="Batches")  # type: ignore[assignment]

            # TRAINING
            if self.p_retain_pool > 0.0:
                x_previous = None

            for i, sample in enumerate(gen):
                x, y = sample

                # Typically, our dataloader supplies a binary, grayscale, RGB or RGBA image.
                # But the NCA operates on multiple hidden channels and output channels, so we
                # need to pad the input image with zeros.
                x = pad_input(x, self.nca, noise=self.nca.pad_noise)
                # Call model-specific input preparation hook
                x = self.nca.prepare_input(x)
                x = x.permute(0, 2, 3, 1)  # --> B W H C

                if (
                    self.p_retain_pool > 0.0
                    and x_previous is not None
                    and np.random.random() <= self.p_retain_pool
                ):
                    # batch sizes might be incompatible if DataLoader has drop_last set to False (default)
                    if x_previous.shape[0] == x.shape[0]:
                        x[:, :, :, self.nca.num_image_channels :] = x_previous[
                            :, :, :, self.nca.num_image_channels :
                        ]

                # batch duplication, slightly stabelizes the training
                x = torch.cat(self.batch_repeat * [x])
                y = torch.cat(self.batch_repeat * [y])

                steps = np.random.randint(*self.steps_range)
                x_pred = train_iteration(
                    x, y, steps, optimizer, scheduler, total_batch_iterations
                )
                if self.p_retain_pool > 0.0:
                    x_previous = x_pred
                total_batch_iterations += 1

            # VISUALIZATION
            if plot_function and summary_writer and (iteration + 1) % save_every == 0:
                figure = plot_function(
                    x.detach().cpu().numpy(),
                    x_pred.detach().cpu().numpy(),
                    y.detach().cpu().numpy(),
                    self.nca,
                )
                summary_writer.add_figure("Training Batch", figure, iteration)

            # VALIDATION
            with torch.no_grad():
                self.nca.eval()
                if self.model_path:
                    torch.save(self.nca.state_dict(), self.model_path)

                if dataloader_val:
                    val_acc = self.nca.validate(
                        dataloader_val,
                        self.steps_validation,
                        iteration,
                        summary_writer,
                    )
                    if val_acc > best_acc:
                        logging.info(
                            f"Accuracy improvement: {best_acc:.5f} --> {val_acc:.5f}"
                        )
                        if best_path:
                            torch.save(self.nca.state_dict(), best_path)
                        best_acc = val_acc
                        best_model = copy.deepcopy(self.nca)
                    if earlystopping is not None:
                        earlystopping.step(val_acc)
        metrics = {}
        if dataloader_test is not None:
            metrics = best_model.metrics(dataloader_test, self.steps_validation)
        return TrainingSummary(best_acc, best_path, metrics)
