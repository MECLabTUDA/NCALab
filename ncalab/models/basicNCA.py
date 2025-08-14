from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

from ..autostepper import AutoStepper
from ..prediction import Prediction
from ..utils import pad_input
from ..visualization import Visual


class BasicNCAModel(nn.Module):
    """
    Abstract base class for NCA models.
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
        autostepper: Optional[AutoStepper] = None,
        use_temporal_encoding: bool = False,
    ):
        """
        :param device [device]: Pytorch device descriptor.
        :param num_image_channels [int]: Number of channels reserved for input image.
        :param num_hidden_channels [int]: Number of hidden channels (communication channels).
        :param num_output_channels [int]: Number of output channels.
        :param fire_rate [float]: Fire rate for stochastic weight update. Defaults to 0.5.
        :param hidden_size [int]: Number of neurons in hidden layer. Defaults to 128.
        :param use_alive_mask [bool]: Whether to use alive masking (channel 3) during training. Defaults to False.
        :param immutable_image_channels [bool]: If image channels should be fixed during inference, which is the case for most segmentation or classification problems. Defaults to True.
        :param num_learned_filters [int]: Number of learned filters. If zero, use two sobel filters instead. Defaults to 2.
        :param filter_padding [str]: Padding type to use. Might affect reliance on spatial cues. Defaults to "circular".
        :param use_laplace [bool]: Whether to use Laplace filter (only if num_learned_filters == 0)
        :param kernel_size [int]: Filter kernel size (only for learned filters)
        :param pad_noise [bool]: Whether to pad input image tensor with noise in hidden / output channels
        :param autostepper [Optional[AutoStepper]]: AutoStepper object to select number of time steps based on activity
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
        self.autostepper = autostepper
        self.use_temporal_encoding = use_temporal_encoding
        self.plot_function = plot_function
        self.validation_metric = validation_metric

        # define input filters
        self._define_filters(num_learned_filters)

        # define model structure
        self.network = self._define_network().to(self.device)

    def _define_network(self):
        input_vector_size = self.num_channels * (self.num_filters + 1)
        if self.use_temporal_encoding:
            input_vector_size += 1
        network = nn.Sequential(
            nn.Conv2d(
                in_channels=input_vector_size,
                out_channels=self.hidden_size,
                bias=True,
                stride=1,
                padding=0,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=self.num_channels,
                bias=False,
                stride=1,
                padding=0,
                kernel_size=1,
            ),
        )
        # initialize final layer with 0
        with torch.no_grad():
            network[-1].weight.data.fill_(0)
        return network

    def _define_filters(self, num_learned_filters: int):
        """
        Define list of perception filters, based on parameters passed in constructor.

        :param num_learned_filters [int]: Number of learned filters in perception filter bank.
        """
        self.filters: list | nn.ModuleList = []
        if num_learned_filters > 0:
            self.num_filters = num_learned_filters
            filters = []
            for _ in range(num_learned_filters):
                filters.append(
                    nn.Conv2d(
                        self.num_channels,
                        self.num_channels,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=(self.kernel_size // 2),
                        padding_mode=self.filter_padding,
                        groups=self.num_channels,
                        bias=False,
                    )
                )
            self.filters = nn.ModuleList(filters).to(self.device)
        else:
            sobel_x = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
            sobel_y = sobel_x.T
            laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            self.filters.extend([sobel_x, sobel_y])
            if self.use_laplace:
                self.filters.append(laplace)
            self.num_filters = len(self.filters)

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

    def _perceive(self, x, step) -> torch.Tensor:
        def _perceive_with(x, weight):
            if isinstance(weight, nn.Conv2d):
                return weight(x)
            # if using a hard coded filter matrix.
            # this is done in the original Growing NCA paper, but learned filters typically
            # work better.
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1, 1, 3, 3).repeat(
                self.num_channels, 1, 1, 1
            )
            return F.conv2d(x, conv_weights, padding=1, groups=self.num_channels)

        perception = [x]
        perception.extend([_perceive_with(x, w) for w in self.filters])
        if self.use_temporal_encoding:
            normalization = 100
            if self.autostepper is not None:
                normalization = self.autostepper.max_steps
            perception.append(
                torch.mul(
                    torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])),
                    step / normalization,
                ).to(self.device)
            )
        dx = torch.cat(perception, 1)
        return dx

    def _update(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        Compute residual cell update.

        :param x [torch.Tensor]: Input tensor, BCWH
        :param step [int]: Current timestep, required for computing temporal encoding.
        """
        assert x.shape[1] == self.num_channels

        # Perception
        dx = self._perceive(x, step)

        # Compute delta from FFNN network
        dx = self.network(dx)

        # Stochastic weight update
        fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), 1, dx.size(2), dx.size(3)]) < fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        if self.immutable_image_channels:
            dx[:, : self.num_image_channels, :, :] *= 0
        return dx

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
        if self.autostepper is None:
            for step in range(steps):
                dx = self._update(x, step)
                x = x + dx

                # Alive masking
                if self.use_alive_mask:
                    life_mask = self._alive(x)
                    life_mask = life_mask
                    x = x.permute(1, 0, 2, 3)  # B C W H --> C B W H
                    x = x * life_mask.float()
                    x = x.permute(1, 0, 2, 3)  # C B W H --> B C W H
            return Prediction(self, steps, x)

        for step in range(self.autostepper.max_steps):
            if self.autostepper.check(step):
                return Prediction(self, step, x)
            # save previous hidden state
            self.autostepper.hidden_i_1 = x[
                :,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
                :,
                :,
            ]
            # single inference time step
            dx = self._update(x, step)
            x = x + dx

            # Alive masking
            if self.use_alive_mask:
                life_mask = self._alive(x)
                life_mask = life_mask
                x = x.permute(1, 0, 2, 3)  # B C W H --> C B W H
                x = x * life_mask.float()
                x = x.permute(1, 0, 2, 3)  # C B W H --> B C W H

            # set current hidden state
            self.autostepper.hidden_i = x[
                :,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
                :,
                :,
            ]
        return Prediction(self, self.autostepper.max_steps, x)

    def loss(self, image: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
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

    def finetune(self):
        """
        Prepare model for fine tuning by freezing everything except the final layer,
        and setting to "train" mode.
        """
        self.train()
        if self.num_learned_filters != 0:
            for filter in self.filters:
                filter.requires_grad_ = False
        for layer in self.network[:-1]:
            layer.requires_grad_ = False

    def metrics(self, pred: torch.Tensor, label: torch.Tensor) -> Dict[str, float]:
        """
        Return dict of standard evaluation metrics.

        :param pred [torch.Tensor]: Predicted image, BCWH.
        :param label [torch.Tensor]: Ground truth label.

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

    def record(self, image: torch.Tensor, steps: int = 100) -> List[Prediction]:
        """
        Record predictions for all time steps and return the resulting
        sequence of predictions.

        :param image [torch.Tensor]: Input image, BCWH.

        :returns [List[Prediction]]: List of Prediction objects.
        """
        assert steps >= 1
        assert image.shape[1] <= self.num_channels
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
        self, image: torch.Tensor, label: torch.Tensor, steps: int
    ) -> Optional[Tuple[Dict[str, float], Prediction]]:
        """
        Make a prediction on an image of the validation set and return metrics computed
        with respect to a labelled validation image.

        :param image [torch.Tensor]: Input image, BCWH
        :param label [torch.Tensor]: Ground truth label
        :param steps [int]: Inference steps

        :returns [Tuple[float, Prediction]]: Validation metric, predicted image BCWH
        """
        prediction = self.predict(image.to(self.device), steps=steps)
        metrics = self.metrics(prediction.output_image, label.to(self.device))
        return metrics, prediction

    def to_dict(self) -> Dict[str, Any]:
        return dict()
