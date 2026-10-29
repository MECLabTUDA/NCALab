from typing import Literal

import numpy as np
import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]
from torch import nn  # type: ignore[import-untyped]

from ncalab.filters import op_laplace, op_prewitt_x, op_prewitt_y
from ncalab.temporalencoding import TemporalEncoding


class BasicNCAPerception(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_channels: int,
        num_learned_filters: int,
        filter_padding: Literal["zero", "reflect", "replicate", "circular"],
        use_laplace: bool,
        temporal_encoding: TemporalEncoding | None = None,
        training_timesteps: int | tuple[int, int] = 100,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.device = device
        self.num_channels = num_channels
        self.num_learned_filters = num_learned_filters
        self.filter_padding = filter_padding
        self.use_laplace = use_laplace
        self.temporal_encoding: TemporalEncoding | None = temporal_encoding
        self.training_timesteps = training_timesteps
        self.kernel_size = kernel_size
        self._define_filters()
        self.to(device)

    def _define_filters(self):
        """
        Define list of perception filters, based on parameters passed in constructor.

        :param num_learned_filters [int]: Number of learned filters in perception filter bank.
        """
        self.filters: list[np.ndarray] | nn.ModuleList = []
        if self.num_learned_filters > 0:
            self.num_filters = self.num_learned_filters
            filters = []
            for _ in range(self.num_learned_filters):
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
            # self.filters.extend([op_sobel_x, op_sobel_y])
            self.filters.extend([op_prewitt_x, op_prewitt_y])
            if self.use_laplace:
                self.filters.append(op_laplace)
            self.num_filters = len(self.filters)

    def vector_dim(self) -> int:
        d = self.num_channels * (len(self.filters) + 1)
        if self.temporal_encoding is not None:
            d += self.temporal_encoding.dim()
        return d

    def perceive(self, x: torch.Tensor, step: int) -> torch.Tensor:
        def _perceive_with(x: torch.Tensor, weight: nn.Conv2d | np.ndarray):
            if isinstance(weight, nn.Conv2d):
                return weight(x)
            # if using a hard coded filter matrix.
            # this is done in the original Growing NCA paper, but learned filters typically
            # work better.
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1, 1, 3, 3).repeat(
                self.num_channels, 1, 1, 1
            )
            x = F.conv2d(x, conv_weights, padding=1, groups=self.num_channels)
            return x

        perception = [x]
        perception.extend([_perceive_with(x, w) for w in self.filters])

        if self.temporal_encoding is not None:
            if type(self.training_timesteps) is int:
                self.temporal_encoding.normalization = self.training_timesteps
            elif type(self.training_timesteps) is tuple:
                self.temporal_encoding.normalization = self.training_timesteps[1]
            perception.append(self.temporal_encoding.vector(step, x).to(self.device))
        dx = torch.cat(perception, 1)
        return dx

    def freeze(self):
        if self.num_learned_filters == 0:
            return
        for filter in self.filters:
            filter.requires_grad_(False)
