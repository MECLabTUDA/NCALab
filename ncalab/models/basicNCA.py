from __future__ import annotations
from typing import Callable, Optional, Dict, Tuple
from os import PathLike

import numpy as np

import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]


class AutoStepper:
    """
    Helps selecting number of timesteps based on NCA activity.
    """

    def __init__(
        self,
        min_steps: int = 10,
        max_steps: int = 100,
        plateau: int = 5,
        verbose: bool = False,
        threshold: float = 1e-2,
    ):
        """
        Constructor.

        :param min_steps [int]: Minimum number of timesteps to always execute. Defaults to 10.
        :param max_steps [int]: Terminate after maximum number of steps. Defaults to 100.
        :param plateau [int]: _description_. Defaults to 5.
        :param verbose [bool]: Whether to communicate. Defaults to False.
         threshold (float, optional): _description_. Defaults to 1e-2.
        """
        assert min_steps >= 1
        assert plateau >= 1
        assert max_steps > min_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.plateau = plateau
        self.verbose = verbose
        self.threshold = threshold


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

        self.plot_function: Callable | None = None
        self.validation_metric = None
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

        self.network = nn.Sequential(
            nn.Linear(
                self.num_channels * (self.num_filters + 1), self.hidden_size, bias=True
            ),
            # nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_channels, bias=False),
        ).to(device)

        # initialize final layer with 0
        with torch.no_grad():
            self.network[-1].weight.zero_()

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

    def perceive(self, x):
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
        y = torch.cat(perception, 1)
        return y

    def update(self, x):
        x = x.permute(0, 3, 1, 2)  # B W H C --> B C W H

        # Perception
        dx = self.perceive(x)

        # Compute delta from FFNN network
        dx = dx.permute(0, 2, 3, 1)  # B C W H --> B W H C
        dx = self.network(dx)

        # Stochastic weight update
        fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) < fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        dx += self.dx_noise * torch.randn([dx.size(0), dx.size(1), dx.size(2), 1]).to(
            self.device
        )

        if self.immutable_image_channels:
            dx[..., : self.num_image_channels] *= 0

        dx = dx.permute(0, 3, 1, 2)  # B W H C --> B C W H
        x = x + dx

        # Alive masking
        if self.use_alive_mask:
            life_mask = self.__alive(x)
            life_mask = life_mask
            x = x.permute(1, 0, 2, 3)  # B C W H --> C B W H
            x = x * life_mask.float()
            x = x.permute(1, 0, 2, 3)  # C B W H --> B C W H
        x = x.permute(0, 2, 3, 1)  # B C W H --> B W H C
        return x

    def forward(
        self,
        x: torch.Tensor,
        steps: int = 1,
        return_steps: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, int]:
        """
        """
        if self.autostepper is None:
            for step in range(steps):
                x = self.update(x)
            if return_steps:
                return x, steps
            return x

        cooldown = 0
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
                    if score >= self.autostepper.threshold:
                        cooldown = 0
                    else:
                        cooldown += 1
                    if cooldown >= self.autostepper.plateau:
                        if self.autostepper.verbose:
                            print(f"Breaking after {step} steps.")
                        if return_steps:
                            return x, step
                        return x
            # save previous hidden state
            hidden_i_1 = x[
                ...,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
            ]
            # single inference time step
            x = self.update(x)
            # set current hidden state
            hidden_i = x[
                ...,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
            ]
        if return_steps:
            return x, self.autostepper.max_steps
        return x

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        :param x: _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
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

    def metrics(self, image, label, steps: int):
        return {}

    def export_onnx(self, path: str | PathLike, optimize: bool = True):
        dummy = torch.zeros((8, 16, 16, self.num_channels)).to(self.device)
        onnx_program = torch.onnx.export(self, dummy, dynamo=True)
        if optimize:
            onnx_program.optimize()
        onnx_program.save(path)

    def validate(
        self,
        image,
        label,
        steps: int,
    ) -> float:
        metrics = self.metrics(image, label, steps)
        return metrics, metrics["prediction"]
