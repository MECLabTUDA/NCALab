from __future__ import annotations
from typing import Callable, Optional, Dict, Tuple

import numpy as np

import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

from ..autostepper import AutoStepper
from ..utils import pad_input


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
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        immutable_image_channels: bool = True,
        num_learned_filters: int = 2,
        dx_noise: float = 0.0,
        filter_padding: str = "circular",
        use_laplace: bool = False,
        kernel_size: int = 3,
        pad_noise: bool = False,
        autostepper: Optional[AutoStepper] = None,
        use_temporal_encoding: bool = False,

    ):
        """
        Constructor.

        :param device [device]: Pytorch device descriptor.
        :param num_image_channels [int]: Number of channels reserved for input image.
        :param num_hidden_channels [int]: Number of hidden channels (communication channels).
        :param num_output_channels [int]: Number of output channels.
        :param fire_rate [float]: Fire rate for stochastic weight update. Defaults to 0.5.
        :param hidden_size [int]: Number of neurons in hidden layer. Defaults to 128.
        :param use_alive_mask [bool]: Whether to use alive masking during training. Defaults to False.
        :param immutable_image_channels [bool]: If image channels should be fixed during inference, which is the case for most segmentation or classification problems. Defaults to True.
        :param num_learned_filters [int]: Number of learned filters. If zero, use two sobel filters instead. Defaults to 2.
        :param dx_noise [float]:
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
        self.dx_noise = dx_noise
        self.pad_noise = pad_noise
        self.autostepper = autostepper
        self.use_temporal_encoding = use_temporal_encoding

        self.plot_function: Optional[Callable] = None
        self.validation_metric: Optional[str] = None
        self.filters: list | nn.ModuleList = []

        if num_learned_filters > 0:
            self.num_filters = num_learned_filters
            filters = []
            for _ in range(num_learned_filters):
                filters.append(
                    nn.Conv2d(
                        self.num_channels,
                        self.num_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size // 2),
                        padding_mode=filter_padding,
                        groups=self.num_channels,
                        bias=False,
                    ).to(self.device)
                )
            self.filters = nn.ModuleList(filters)
        else:
            sobel_x = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
            sobel_y = sobel_x.T
            laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            self.filters.append(sobel_x)
            self.filters.append(sobel_y)
            if self.use_laplace:
                self.filters.append(laplace)
            self.num_filters = len(self.filters)

        input_vector_size = self.num_channels * (self.num_filters + 1)
        if self.use_temporal_encoding:
            input_vector_size += 1
        self.network = nn.Sequential(
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
        ).to(device)

        # initialize final layer with 0
        with torch.no_grad():
            self.network[-1].weight.data.fill_(0)

        self.meta: dict = {}

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input. Intended to be overwritten by subclass, if preprocessing
        is necessary.

        :param x [torch.Tensor]: Input tensor to preprocess.

        :returns: Processed tensor.
        """
        return x

    def __alive(self, x):
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
            perception.append(
                torch.mul(
                    torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])), step / 100
                ).to(self.device)
            )
        dx = torch.cat(perception, 1)
        return dx

    def _update(self, x: torch.Tensor, step):
        """
        Compute residual cell update.

        :param x [torch.Tensor]: Input tensor, BCWH
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
        return_steps: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, int]:
        """
        :param x [torch.Tensor]: Input image, padded along the channel dimension, BCWH.
        :param steps [int]: Time steps in forward pass.
        :param return_steps [bool]: Whether to return number of steps we took.

        :returns: Output image, BWHC
        """
        if self.autostepper is None:
            for step in range(steps):
                dx = self._update(x, step)
                x = x + dx
            return x, steps

        # invariant: auto_min_steps > 0, so both of these will be defined when used
        hidden_i: torch.Tensor | None = None
        hidden_i_1: torch.Tensor | None = None
        for step in range(self.autostepper.max_steps):
            with torch.no_grad():
                if (
                    step >= self.autostepper.min_steps
                    and hidden_i is not None
                    and hidden_i_1 is not None
                ):
                    # normalized absolute difference between two hidden states
                    score = (hidden_i - hidden_i_1).abs().sum() / (
                        hidden_i.shape[0]
                        * hidden_i.shape[1]
                        * hidden_i.shape[2]
                        * hidden_i.shape[3]
                    )
                    if self.autostepper.check(step, score):
                        return x, step
            # save previous hidden state
            hidden_i_1 = x[
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
                life_mask = self.__alive(x)
                life_mask = life_mask
                x = x.permute(1, 0, 2, 3)  # B C W H --> C B W H
                x = x * life_mask.float()
                x = x.permute(1, 0, 2, 3)  # C B W H --> B C W H
            # set current hidden state
            hidden_i = x[
                :,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
                :,
                :,
            ]
        return x, self.autostepper.max_steps

    def loss(self, image: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        Needs to be overloaded by any subclass.

        :param image [torch.Tensor]: Input image, BWHC.
        :param label [torch.Tensor]: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        return NotImplemented

    def get_meta_dict(self) -> dict:
        return dict(
            device=str(self.device),
            num_image_channels=self.num_image_channels,
            num_hidden_channels=self.num_hidden_channels,
            num_output_channels=self.num_output_channels,
            fire_rate=self.fire_rate,
            hidden_size=self.hidden_size,
            use_alive_mask=self.use_alive_mask,
            immutable_image_channels=self.immutable_image_channels,
            num_learned_filters=self.num_learned_filters,
            dx_noise=self.dx_noise,
            **self.meta,
        )

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
        Needs to include special item 'prediction', containing the predicted image (all channels).

        :param pred [torch.Tensor]: Predicted image, BWHC.
        :param label [torch.Tensor]: Ground truth label.

        :returns [Dict]: Dict of metrics, mapped by their names.
        """
        return {}

    def predict(self, image: torch.Tensor, steps: int = 100) -> torch.Tensor:
        """
        :param image [torch.Tensor]: Input image, BCWH.

        :returns [torch.Tensor]: Output image, BWHC
        """
        assert steps >= 1
        assert image.shape[1] <= self.num_channels
        self.eval()
        with torch.no_grad():
            x = image.clone()
            x = pad_input(x, self, noise=self.pad_noise)
            x = self.prepare_input(x)
            x = self.forward(x, steps=steps)  # type: ignore[assignment]
            return x

    def validate(
        self, image: torch.Tensor, label: torch.Tensor, steps: int
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        :param image [torch.Tensor]: Input image, BCWH
        :param label [torch.Tensor]: Ground truth label
        :param steps [int]: Inference steps

        :returns [Tuple[float, torch.Tensor]]: Validation metric, predicted image BWHC
        """
        pred = self.predict(image.to(self.device), steps=steps)
        metrics = self.metrics(pred, label.to(self.device))
        return metrics, pred
