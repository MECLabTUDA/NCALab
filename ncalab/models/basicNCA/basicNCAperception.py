from typing import TYPE_CHECKING, List

import numpy as np
import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from . import BasicNCAModel


class BasicNCAPerception:
    def __init__(self, nca: "BasicNCAModel"):
        super().__init__()
        self.nca = nca
        self._define_filters()

    def _define_filters(self):
        """
        Define list of perception filters, based on parameters passed in constructor.

        :param num_learned_filters [int]: Number of learned filters in perception filter bank.
        """
        self.filters: List[np.ndarray] | nn.ModuleList = []
        if self.nca.num_learned_filters > 0:
            self.num_filters = self.nca.num_learned_filters
            filters = []
            for _ in range(self.nca.num_learned_filters):
                filters.append(
                    nn.Conv2d(
                        self.nca.num_channels,
                        self.nca.num_channels,
                        kernel_size=self.nca.kernel_size,
                        stride=1,
                        padding=(self.nca.kernel_size // 2),
                        padding_mode=self.nca.filter_padding,
                        groups=self.nca.num_channels,
                        bias=False,
                    )
                )
            self.filters = nn.ModuleList(filters).to(self.nca.device)
        else:
            sobel_x = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
            sobel_y = sobel_x.T
            laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            self.filters.extend([sobel_x, sobel_y])
            if self.nca.use_laplace:
                self.filters.append(laplace)
            self.num_filters = len(self.filters)

    def perceive(self, x, step: int) -> torch.Tensor:
        def _perceive_with(x, weight):
            if isinstance(weight, nn.Conv2d):
                return weight(x)
            # if using a hard coded filter matrix.
            # this is done in the original Growing NCA paper, but learned filters typically
            # work better.
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(
                self.nca.device
            )
            conv_weights = conv_weights.view(1, 1, 3, 3).repeat(
                self.nca.num_channels, 1, 1, 1
            )
            return F.conv2d(x, conv_weights, padding=1, groups=self.nca.num_channels)

        perception = [x]
        perception.extend([_perceive_with(x, w) for w in self.filters])
        if self.nca.use_temporal_encoding:
            normalization: int = 1
            if type(self.nca.training_timesteps) is int:
                self.nca.training_timesteps  # maximum steps
            elif type(self.nca.training_timesteps) is tuple:
                 self.nca.training_timesteps[1]
            perception.append(
                torch.mul(
                    torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])),
                    step / normalization,
                ).to(self.nca.device)
            )
        dx = torch.cat(perception, 1)
        return dx

    def freeze(self):
        if self.nca.num_learned_filters == 0:
            return
        for filter in self.filters:
            filter.requires_grad_(False)
