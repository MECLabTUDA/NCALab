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
        learned_filters=2,
    ):
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

        self.hidden_layer = nn.Linear(
            self.num_channels * (self.num_filters + 1), hidden_size
        ).to(device)
        self.activation = nn.ReLU().to(device)
        self.final_layer = nn.Linear(hidden_size, self.num_channels, bias=False).to(device)
        with torch.no_grad():
            self.final_layer.weight.zero_()

    def alive(self, x):
        if not self.use_alive_mask:
            return True
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x):
        def _perceive_with(x, weight):
            if type(weight) == nn.Conv2d:
                return weight(x)
            # hard coded filter matrix:
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1, 1, 3, 3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        perception = [x]
        perception.extend([_perceive_with(x, w) for w in self.filters])
        y = torch.cat(perception, 1)
        return y

    def update(self, x, angle):
        x = x.transpose(1, 3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x)
        dx = dx.transpose(1, 3)
        dx = self.hidden_layer(dx)
        dx = self.activation(dx)
        dx = self.final_layer(dx)

        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > self.fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic
        dx[..., :self.num_image_channels] *= 0
        
        x = x + dx.transpose(1, 3)
        x = x.transpose(1, 0)
        life_mask = x[0] > 0
        x = x * life_mask
        x = x.transpose(0, 1)
        x = x.transpose(1, 3)
        return x

    def forward(self, x, steps=1, angle=0.0):
        for _ in range(steps):
            x = self.update(x, angle)
        return x
