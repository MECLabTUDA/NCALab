import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BasicNCAModel(nn.Module):
    def __init__(
        self,
        device,
        num_image_channels: int,
        num_hidden_channels: int,
        num_output_channels: int,
        fire_rate=0.5,
        hidden_size=128,
        use_alive_mask=False,
        immutable_image_channels=True,
        learned_filters=0,
    ):
        super(BasicNCAModel, self).__init__()

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

        if learned_filters > 0:
            self.num_filters = learned_filters
            for _ in range(learned_filters):
                self.filters.append(...)
        else:
            self.num_filters = 2
            self.filters.append(
                np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
            )
            self.filters.append(
                (np.outer([1, 2, 1], [-1, 0, 1]) / 8.0).T
            )

        self.hidden_layer = nn.Linear(self.num_channels * (self.num_filters + 1), hidden_size)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, self.num_channels, bias=False)
        with torch.no_grad():
            self.output_layer.weight.zero_()

        self.to(device)

    def alive(self, x):
        if not self.use_alive_mask:
            return True
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x):
        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        perception = [x]
        perception.extend([_perceive_with(x, w) for w in self.filters])
        y = torch.cat(perception, 1)
        return y

    def update(self, x, angle):
        x = x.permute(1, 3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.permute(1, 3)
        dx = self.hidden_layer(dx)
        dx = self.activation(dx)
        dx = self.output_layer(dx)

        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > self.fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        if self.immutable_image_channels:
            dx[..., :self.num_image_channels] = 0

        x = x + dx.permute(1, 3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        if self.immutable_image_channels:
            life_mask[..., :self.num_image_channels] = 1
        x = x * life_mask
        return x.permute(1, 3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for _ in range(steps):
            x = self.update(x, fire_rate, angle)
        return x
