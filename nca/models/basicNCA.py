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
        fire_rate=0.5,
        hidden_size=128,
        use_alive_mask=False,
        immutable_image_channels=True,
        learned_filters=2
    ):
        """_summary_

        Args:
            device (device): _description_
            num_image_channels (int): _description_
            num_hidden_channels (int): _description_
            num_output_channels (int): _description_
            fire_rate (float, optional): _description_. Defaults to 0.5.
            hidden_size (int, optional): _description_. Defaults to 128.
            use_alive_mask (bool, optional): _description_. Defaults to False.
            immutable_image_channels (bool, optional): _description_. Defaults to True.
            learned_filters (int, optional): _description_. Defaults to 2.
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
        self.learned_filters = learned_filters
        self.filters = []

        self.plot_function = None

        if learned_filters > 0:
            self.num_filters = learned_filters
            for _ in range(learned_filters):
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
        else:
            self.num_filters = 2
            self.filters.append(np.outer([1, 2, 1], [-1, 0, 1]) / 8.0)
            self.filters.append((np.outer([1, 2, 1], [-1, 0, 1]) / 8.0).T)

        self.network = nn.Sequential(
            nn.Linear(self.num_channels * (self.num_filters + 1), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_channels, bias=False)
        ).to(device)

        # initialize final layer with 0
        with torch.no_grad():
            self.network[-1].weight.zero_()

    def alive(self, x):
        mask = F.max_pool2d(x[:, self.num_image_channels, :, :], kernel_size=3, stride=1, padding=1) > 0.1
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

    def update(self, x):
        x = x.transpose(1, 3)
        pre_life_mask = self.alive(x)

        # Perception
        dx = self.perceive(x)

        # Compute delta from FFNN network
        dx = dx.transpose(1, 3)
        dx = self.network(dx)

        # Stochastic weight update
        stochastic = (
            torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > self.fire_rate
        )
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        if self.immutable_image_channels:
            dx[..., : self.num_image_channels] *= 0

        # Alive masking
        x = x + dx.transpose(1, 3) # B W H C --> B C W H
        # FIXME: Something is wrong in the state of Denmark
        if self.use_alive_mask:
            x = x.transpose(0, 1)
            life_mask = self.alive(x) & pre_life_mask
            x = x * life_mask.float()
            x = x.transpose(1, 0)
        x = x.transpose(1, 3) # B C W H --> B W H C
        return x

    def forward(self, x, steps: int = 1):
        for _ in range(steps):
            x = self.update(x)
        return x

    def loss(self, x, target):
        return NotImplemented
