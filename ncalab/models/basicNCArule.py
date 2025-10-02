import torch
from torch import nn


class BasicNCARule(nn.Module):
    def __init__(
        self,
        device: torch.device,
        input_size: int,
        hidden_size: int,
        output_size: int,
        nonlinearity: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self._build_network()
        self._initialize_network()
        self.network.to(self.device)

    def _build_network(self):
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_size,
                out_channels=self.hidden_size,
                bias=True,
                stride=1,
                padding=0,
                kernel_size=1,
            ),
            self.nonlinearity(),
            nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=self.output_size,
                bias=False,
                stride=1,
                padding=0,
                kernel_size=1,
            ),
        )

    def _initialize_network(self):
        with torch.no_grad():
            data = self.network[-1].weight
            assert type(data) is torch.nn.parameter.Parameter
            data.fill_(0)

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def freeze(self, freeze_last: bool = False):
        layers = self.network
        if not freeze_last:
            layers = self.network[:-1]
        for layer in layers:
            layer.requires_grad_(False)
