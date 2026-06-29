from __future__ import annotations

from copy import copy
from pathlib import Path, PosixPath  # for type hint
from typing import Dict, Optional, Tuple

import numpy as np
import torch  # type: ignore[import-untyped]
import torch.optim as optim  # type: ignore[import-untyped]
from torch.utils.data import DataLoader  # for type hint
from torch.utils.tensorboard import SummaryWriter  # for type hint
from tqdm import tqdm  # type: ignore[import-untyped]

from ..models.basicNCA import AbstractNCAModel  # for type hint
from ..prediction import Prediction
from ..utils import interpret_range_parameter, pad_input
from ..visualization import Visual
from .earlystopping import EarlyStopping
from .pool import Pool
from .traininghistory import TrainingHistory


class BasicNCATrainer:
    """
    Trainer class for any model subclassing BasicNCA.
    """

    def __init__(
        self,
        nca: AbstractNCAModel,
        model_path: Optional[Path | PosixPath],
        gradient_clipping: bool = False,
        optimizer: Optional[optim.Optimizer] = None,
        lr: float = 1e-3,
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        batch_repeat: int = 2,
        max_epochs: int = 200,
        pool: Optional[Pool] = None,
    ):
        """
        :param nca: NCA model instance to train.
        :type nca: ncalab.AbstractNCAModel
        :param model_path: Path to saved models. If None, models are not saved, defaults to None.
        :type model_path: Path | PosixPath, optional
        :param gradient_clipping: Whether to clip gradients, defaults to False.
        :type gradient_clipping: bool, optional
        :param lr: Initial learning rate, defaults to 16e-4.
        :type lr: float, optional
        :param batch_repeat: How often each batch will be duplicated, dfaults to 2.
        :param max_epochs: Maximum number of epochs in training, defaults to 200.
        :param pool: Sample pool object.
        :type pool: ncalab.Pool
        """
        assert batch_repeat >= 1
        assert max_epochs > 0

        self.nca = nca
        self.model_path = model_path
        self.gradient_clipping = gradient_clipping
        self.batch_repeat = batch_repeat
        self.max_epochs = max_epochs
        self.lr = lr
        self.optimizer = (
            optim.Adam(self.nca.parameters(), self.lr, betas=(0.9, 0.95))
            if optimizer is None
            else optimizer
        )
        self.pool = pool
        self.lr_scheduler = lr_scheduler

    def info(self) -> str:
        """
        Shows a markdown-formatted info string with training parameters.
        Useful for showing info on tensorboard to keep track of parameter changes.

        :return: Markdown-formatted info string.
        :rtype: str
        """
        s = "BasicNCATrainer Info\n"
        s += "-------------------\n"
        for attribute in (
            "model_path",
            "gradient_clipping",
            "batch_repeat",
            "max_epochs",
        ):
            attribute_f = attribute.title().replace("_", " ")
            s += f"**{attribute_f}:** {getattr(self, attribute)}\n"
        return s

    def _train_iteration(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        head_optimizer: torch.optim.Optimizer | None,
        total_batch_iterations: int,
        summary_writer: Optional[SummaryWriter] = None,
    ) -> Tuple[Prediction, Dict[str, torch.Tensor]]:
        """
        Run a single training iteration.

        :param x: Input training images.
        :param y: Input training labels.
        :param steps: Number of NCA inference time steps.
        :param optimizer: Optimizer.
        :param total_batch_iterations: Total training batch iterations
        :type total_batch_iterations: int
        :param summary_writer: Tensorboard SummaryWriter
        :type summary_writer: SummaryWriter, optional

        :returns: Predicted image.
        :rtype: Tuple[Prediction, Dict[str, torch.Tensor]]
        """
        device = self.nca.device
        self.nca.train()
        self.optimizer.zero_grad()
        if head_optimizer is not None:
            head_optimizer.zero_grad()
        x_in = x.clone().to(self.nca.device)
        x_in = pad_input(x_in, self.nca, noise=self.nca.pad_noise)
        prediction = self.nca(
            x_in, steps=interpret_range_parameter(self.nca.training_timesteps)
        )
        losses = self.nca.loss(prediction, y.to(device))
        losses["total"].backward()

        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.nca.parameters(), 1.0)
        self.optimizer.step()
        if head_optimizer is not None:
            head_optimizer.step()
        if summary_writer:
            for key in losses:
                summary_writer.add_scalar(
                    f"Loss/train_{key}", losses[key], total_batch_iterations
                )
        return prediction, losses

    def train(
        self,
        dataloader_train: DataLoader,
        dataloader_val: Optional[DataLoader] = None,
        dataloader_test: Optional[DataLoader] = None,
        save_every: Optional[int] = None,
        summary_writer: Optional[SummaryWriter] = None,
        plot_function: Optional[Visual] = None,
        earlystopping: Optional[EarlyStopping] = None,
    ) -> TrainingHistory:
        """
        Execute basic NCA training loop with a single function call.

        :param dataloader_train: Training DataLoader
        :type dataloader_train: DataLoader
        :param dataloader_val: Validation DataLoader, defaults to None
        :type dataloader_val: Optional[DataLoader], optional
        :param dataloader_test: Test DataLoader for final evaluation after successful training, defaults to None
        :type dataloader_test: Optional[DataLoader], optional
        :param save_every: How often to save model state (in epochs). Setting > 1 is useful for very small datasets, like growing lizard., defaults to None
        :type save_every: Optional[int], optional
        :param summary_writer: Tensorboard SummaryWriter, defaults to None
        :type summary_writer: Optional[SummaryWriter], optional
        :param plot_function: Plot function override. If None, use model's default, defaults to None
        :type plot_function: Optional[Visual], optional
        :param earlystopping: EarlyStopping object, defaults to None
        :type earlystopping: Optional[EarlyStopping], optional
        :return: TrainingHistory object, populated with best and final model checkpoints etc.
        :rtype: TrainingHistory
        """
        history = TrainingHistory(self.model_path, {}, 0, self.nca)

        if save_every is None:
            save_every = 1
        assert save_every > 0

        # Use default plot function for NCA flavor if no override is explicitly given
        if not plot_function:
            if self.nca.plot_function:
                plot_function = self.nca.plot_function

        if summary_writer is not None:
            summary_writer.add_text("Training Info", self.info())

        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9)
        head_optimizer = None
        if self.nca.head is not None:
            if self.nca.head.optimizer is not None:
                head_optimizer = copy(self.nca.head.optimizer)

        # MAIN LOOP
        total_batch_iterations = 0
        train_prediction = None
        x = None
        y = None
        for iteration in tqdm(range(self.max_epochs), desc="Epochs"):
            if earlystopping is not None:
                if earlystopping.done():
                    break

            # disable tqdm progress bar if dataset has only one sample, e.g. in the growing task
            gen = iter(dataloader_train)
            if len(dataloader_train) > 3:
                gen = tqdm(gen, desc="Batches")  # type: ignore[assignment]

            all_losses = []
            # TRAINING
            for sample in gen:
                x, y = sample  # x: BCWH, y: BWHC
                if len(y.shape) == 4:
                    y = y.permute(0, 3, 1, 2)  # BWHC --> BCWH

                # Typically, our dataloader supplies a binary, grayscale, RGB or RGBA image.
                # But the NCA operates on multiple hidden channels and output channels, so we
                # need to pad the input image with zeros.
                x = pad_input(x, self.nca, noise=self.nca.pad_noise)
                # Call model-specific input preparation hook
                x = self.nca.prepare_input(x)
                if self.pool is not None:
                    x = self.pool.sample(x)

                # Batch duplication, slightly stabelizes the training
                if self.batch_repeat > 1:
                    x = torch.cat(self.batch_repeat * [x])
                    y = torch.cat(self.batch_repeat * [y])

                train_prediction, losses = self._train_iteration(
                    x,
                    y,
                    head_optimizer,
                    total_batch_iterations,
                    summary_writer,
                )
                assert "total" in losses, "Model: Loss dict must contain 'total' item."
                with torch.no_grad():
                    total_batch_iterations += 1
                    if self.pool is not None:
                        self.pool.update(train_prediction.output_image)
                    all_losses.append(losses["total"].item())
            history.loss.append(float(np.mean(all_losses)))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            with torch.no_grad():
                self.nca.eval()
                # VISUALIZATION
                if (
                    plot_function
                    and summary_writer
                    and (iteration + 1) % save_every == 0
                    and train_prediction is not None
                    and x is not None
                    and y is not None
                ):
                    figure = plot_function.show(
                        self.nca,
                        x.detach().cpu().numpy(),
                        train_prediction,
                        y.detach().cpu().numpy(),
                    )
                    summary_writer.add_figure("Training Batch", figure, iteration)

                # VALIDATION
                if dataloader_val is not None:
                    metrics, _ = self.nca.validate(dataloader_val)
                    for name, value in metrics.items():
                        if summary_writer is not None:
                            summary_writer.add_scalar(
                                f"Acc/Val/{name}", value, iteration
                            )
                    val_acc = 0.0
                    if self.nca.validation_metric in metrics:
                        val_acc = metrics.get(self.nca.validation_metric, 0.0)
                        if earlystopping is not None:
                            earlystopping.step(val_acc)
                    history.update(iteration, self.nca, val_acc)

                    # Visualize validation batch
                    if (
                        plot_function
                        and summary_writer
                        and (iteration + 1) % save_every == 0
                    ):
                        x, y = next(iter(dataloader_val))
                        val_prediction = self.nca.predict(x.to(self.nca.device))
                        figure = plot_function.show(
                            self.nca,
                            x.detach().cpu().numpy(),
                            val_prediction,
                            y.detach().cpu().numpy(),
                        )
                        summary_writer.add_figure("Validation Batch", figure, iteration)
                else:
                    history.update(iteration, self.nca, 0, overwrite=True)
                if (iteration + 1) % save_every == 0:
                    history.save()
        # After training: Compute metrics on test set for training summary
        with torch.no_grad():
            history.metrics = {}
            if dataloader_test is not None and history.best_model is not None:
                metrics, _ = history.best_model.validate(dataloader_test)
                history.metrics.update(metrics)
        return history
