from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

from ..prediction import Prediction
from ..utils import intepret_range_parameter, pad_input, unwrap
from ..visualization import Visual
from .basicNCAhead import BasicNCAHead
from .basicNCAperception import BasicNCAPerception
from .basicNCArule import BasicNCARule


class BasicNCAModel(nn.Module):
    """
    Abstract base class for NCA models.

    BasicNCAModel is a composition of an NCA backbone model (called "rule"), and
    an (optional) head module for downstream tasks.
    """

    def __init__(
        self,
        device: torch.device,
        num_image_channels: int,
        num_hidden_channels: int,
        num_output_channels: int,
        plot_function: Optional[Visual] = None,
        validation_metric: Optional[str] = None,
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        immutable_image_channels: bool = True,
        num_learned_filters: int = 2,
        filter_padding: str = "reflect",
        use_laplace: bool = False,
        kernel_size: int = 3,
        pad_noise: bool = False,
        use_temporal_encoding: bool = False,
        rule_type: type[BasicNCARule] = BasicNCARule,
        training_timesteps: int | Tuple[int, int] = 100,
        inference_timesteps: int | Tuple[int, int] = 100,
    ):
        """
        :param device: Pytorch device descriptor.
        :param num_image_channels: Number of channels reserved for input image.
        :param num_hidden_channels: Number of hidden channels (communication channels).
        :param num_output_channels: Number of output channels.
        :param fire_rate: Fire rate for stochastic weight update. Defaults to 0.5.
        :param hidden_size: Number of neurons in hidden layer. Defaults to 128.
        :param use_alive_mask: Whether to use alive masking (channel 3) during training. Defaults to False.
        :param immutable_image_channels: If image channels should be fixed during inference, which is the case for most segmentation or classification problems. Defaults to True.
        :param num_learned_filters: Number of learned filters. If zero, use two sobel filters instead. Defaults to 2.
        :param filter_padding: Padding type to use. Might affect reliance on spatial cues. Defaults to "circular".
        :param use_laplace: Whether to use Laplace filter (only if num_learned_filters == 0)
        :param kernel_size: Filter kernel size (only for learned filters)
        :param pad_noise: Whether to pad input image tensor with noise in hidden / output channels
        """
        super(BasicNCAModel, self).__init__()

        self.device = device
        self.to(device)

        self.num_image_channels = num_image_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.num_channels = (
            num_image_channels + num_hidden_channels + num_output_channels
        )
        self.fire_rate = fire_rate
        self.hidden_size = hidden_size
        self.use_alive_mask = use_alive_mask
        self.immutable_image_channels = immutable_image_channels
        self.num_learned_filters = num_learned_filters
        self.use_laplace = use_laplace
        self.kernel_size = kernel_size
        self.filter_padding = filter_padding
        self.pad_noise = pad_noise
        self.use_temporal_encoding = use_temporal_encoding
        self.plot_function = plot_function
        self.validation_metric = validation_metric
        self.training_timesteps = training_timesteps
        self.inference_timesteps = inference_timesteps

        # define model structure
        # perception
        self.perception = BasicNCAPerception(self)
        self.input_vector_size = self.num_channels * (self.perception.num_filters + 1)
        if self.use_temporal_encoding:
            self.input_vector_size += 1
        # rule
        self.rule_type = rule_type
        self.rule = self._define_rule()
        self.head: BasicNCAHead | None = None
        # pre-compute stochastic weight update
        self._stochastic: torch.Tensor | None = None

    def _define_rule(self):
        return self.rule_type(
            self.device, self.input_vector_size, self.hidden_size, self.num_channels
        )

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input. Intended to be overwritten by subclass, if preprocessing
        is necessary.

        :param x [torch.Tensor]: Input tensor to preprocess.

        :returns: Processed tensor.
        """
        return x

    def _alive(self, x):
        mask = (
            F.max_pool2d(
                x[:, 3, :, :],
                kernel_size=3,
                stride=1,
                padding=1,
            )
            > 0.1
        )
        return mask

    def _update(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        Compute residual cell update.

        :param x [torch.Tensor]: Input tensor, BCWH
        :param step [int]: Current timestep, required for computing temporal encoding.
        """
        assert x.shape[1] == self.num_channels

        # Perception
        dx = self.perception.perceive(x, step)

        # Compute delta from FFNN network
        dx = self.rule(dx)

        # Stochastic weight update
        dx = dx * unwrap(self._stochastic)[step % len(unwrap(self._stochastic))]

        if self.immutable_image_channels:
            dx[:, : self.num_image_channels, :, :] *= 0
        return dx

    def _forward_step(self, x: torch.Tensor, step: int):
        dx = self._update(x, step)
        x = x + dx

        # Alive masking
        if self.use_alive_mask:
            life_mask = self._alive(x)
            life_mask = life_mask
            x = x.permute(1, 0, 2, 3)  # B C W H --> C B W H
            x = x * life_mask.float()
            x = x.permute(1, 0, 2, 3)  # C B W H --> B C W H
        return x

    def forward(
        self,
        x: torch.Tensor,
        steps: int = 1,
    ) -> Prediction:
        """
        :param x [torch.Tensor]: Input image, padded along the channel dimension, BCWH.
        :param steps [int]: Time steps in forward pass.

        :returns [Prediction]: Prediction object.
        """
        assert x.shape[1] == self.num_channels
        S = torch.rand([steps, 1, 1, x.size(2), x.size(3)]) < self.fire_rate
        self._stochastic = S.float().to(self.device)
        for step in range(steps):
            x = self._forward_step(x, step)

        if self.head is not None:
            hidden = x[
                :,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
                :,
                :,
            ]
            head_prediction = self.head(hidden)
            return Prediction(self, steps, x, head_prediction)
        return Prediction(self, steps, x)

    def loss(self, pred: Prediction, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss. Needs to be overloaded by any subclass.
        Please note that the returned dict needs to hold "total" key in which the
        total loss is stored, which is typically a weighted sum of other losses.
        The total loss is backpropagated, whereas the other losses are sent to
        tensorboard.

        :param image [torch.Tensor]: Input image, BCWH.
        :param label [torch.Tensor]: Ground truth, BCWH.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        return NotImplemented

    def finetune(self, freeze_head: bool = False):
        """
        Prepare model for fine tuning by freezing everything except the final layer,
        and setting to "train" mode.
        """
        self.train()
        self.perception.freeze()
        self.rule.freeze()
        if freeze_head and self.head is not None:
            self.head.freeze()

    def metrics(self, pred: Prediction, label: torch.Tensor) -> Dict[str, float]:
        """
        Return dict of standard evaluation metrics.

        :param pred: Prediction.
        :type pred: Prediction
        :param label: Ground truth label.

        :returns [Dict]: Dict of metrics, mapped by their names.
        """
        return {}

    def predict(self, image: torch.Tensor, steps: int = 100) -> Prediction:
        """
        :param image [torch.Tensor]: Input image, BCWH.

        :returns [Prediction]: Prediction object.
        """
        assert steps >= 1
        assert image.shape[1] <= self.num_channels
        self.eval()
        with torch.no_grad():
            x = image.clone()
            x = pad_input(x, self, noise=self.pad_noise)
            x = self.prepare_input(x)
            prediction = self.forward(x, steps=steps)
            return prediction

    def record(
        self, image: torch.Tensor, steps: Optional[int] = None
    ) -> List[Prediction]:
        """
        Record predictions for all time steps and return the resulting
        sequence of predictions.

        :param image [torch.Tensor]: Input image, BCWH.

        :returns [List[Prediction]]: List of Prediction objects.
        """
        assert image.shape[1] <= self.num_channels
        if steps is None:
            steps = intepret_range_parameter(self.inference_timesteps)
        self.eval()
        sequence = []
        with torch.no_grad():
            x = image.clone()
            x = pad_input(x, self, noise=self.pad_noise)
            x = self.prepare_input(x)
            for _ in range(steps):
                prediction = self.forward(x, steps=1)
                sequence.append(prediction)
                x = prediction.output_image
            return sequence

    def validate(
        self, image: torch.Tensor, label: torch.Tensor, steps: Optional[int] = None
    ) -> Optional[Tuple[Dict[str, float], Prediction]]:
        """
        Make a prediction on an image of the validation set and return metrics computed
        with respect to a labelled validation image.

        :param image [torch.Tensor]: Input image, BCWH
        :param label [torch.Tensor]: Ground truth label
        :param steps [int]: Inference steps

        :returns [Tuple[float, Prediction]]: Validation metric, predicted image BCWH
        """
        if steps is None:
            steps = intepret_range_parameter(self.inference_timesteps)
        prediction = self.predict(image.to(self.device), steps=steps)
        metrics = self.metrics(prediction, label.to(self.device))
        return metrics, prediction

    def to_dict(self) -> Dict[str, Any]:
        return dict()

    def num_trainable_parameters(self) -> int:
        """
        Returns the number of trainable model parameters.

        :return: Number of trainable parameters.
        :rtype: int
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return int(sum([np.prod(p.size()) for p in model_parameters]))
