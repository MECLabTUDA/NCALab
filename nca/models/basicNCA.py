import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicNCAModel(nn.Module):
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
    ):
        """Basic abstract class for NCA models.

        Args:
            device (device): Pytorch device descriptor.
            num_image_channels (int): Number of channels reserved for input image.
            num_hidden_channels (int): Number of hidden channels (communication channels).
            num_output_channels (int): Number of output channels.
            fire_rate (float, optional): Fire rate for stochastic weight update. Defaults to 0.5.
            hidden_size (int, optional): Number of neurons in hidden layer. Defaults to 128.
            use_alive_mask (bool, optional): Whether to use alive masking during training. Defaults to False.
            immutable_image_channels (bool, optional): If image channels should be fixed during inference,
                which is the case for most segmentation or classification problems. Defaults to True.
            learned_filters (int, optional): Number of learned filters. If zero, use two sobel filters instead. Defaults to 2.
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
        self.use_alive_mask = use_alive_mask
        self.immutable_image_channels = immutable_image_channels
        self.num_learned_filters = num_learned_filters
        self.filters = []

        self.hidden_size = hidden_size

        self.plot_function = None

        if num_learned_filters > 0:
            self.num_filters = num_learned_filters
            for _ in range(num_learned_filters):
                self.filters.append(
                    nn.Conv2d(
                        self.num_channels,
                        self.num_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode="reflect",
                        groups=self.num_channels,
                        bias=False,
                    ).to(self.device)
                )
                self.filters = nn.ModuleList(self.filters)
        else:
            self.num_filters = 2
            self.filters.append(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0)
            self.filters.append((np.outer([1, 2, 1], [-1, 0, 1]) / 8.0).T)

        self.network = nn.Sequential(
            nn.Linear(self.num_channels * (self.num_filters + 1), hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_channels, bias=False),
        ).to(device)

        # initialize final layer with 0
        with torch.no_grad():
            self.network[-1].weight.zero_()

        self.meta = {}

    def alive(self, x):
        mask = (
            F.max_pool2d(
                x[
                    :,
                    self.num_image_channels : self.num_image_channels
                    + self.num_hidden_channels,
                    :,
                    :,
                ],
                kernel_size=3,
                stride=1,
                padding=1,
            )
            > 0.1
        )
        mask = torch.any(mask, dim=1)
        return mask

    def perceive(self, x):
        def _perceive_with(x, weight):
            if type(weight) == nn.Conv2d:
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

    def update(self, x, step=0):
        x = x.transpose(1, 3)  # B W H C --> B C W H

        hidden_channels = x[
            :,
            self.num_image_channels : self.num_image_channels
            + self.num_hidden_channels,
            :,
            :,
        ]
        pre_life_mask = self.alive(hidden_channels)

        # Perception
        dx = self.perceive(x)

        # Compute delta from FFNN network
        dx = dx.transpose(1, 3)
        dx = self.network(dx)

        # Stochastic weight update
        fire_rate = self.fire_rate
        stochastic = (
            torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        )
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        if self.immutable_image_channels:
            dx[..., : self.num_image_channels] *= 0

        x = x + dx.transpose(1, 3)  # B W H C --> B C W H

        # Alive masking
        life_mask = self.alive(hidden_channels)
        life_mask = life_mask & pre_life_mask

        if self.use_alive_mask:
            x = x.transpose(0, 1)  # B C W H --> C B W H
            x = x * life_mask.float()
            x = x.transpose(1, 0)  # C B W H --> B C W H
        x = x.transpose(1, 3)  # B C W H --> B W H C
        return x

    def forward(self, x, steps: int = 1):
        for step in range(steps):
            x = self.update(x, step)
        return x

    def loss(self, x, target):
        """_summary_

        Args:
            x (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        return NotImplemented

    def validate(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        steps: int,
        batch_iteration: int,
        summary_writer=None,
    ):
        """_summary_

        Args:
            x (torch.Tensor): _description_
            target (torch.Tensor): _description_
            steps (int): _description_
            batch_iteration (int): _description_
            summary_writer (_type_, optional): _description_. Defaults to None.

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
            **self.meta
        )
