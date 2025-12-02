import torch
from torch import nn

from ncalab.models.basicNCA.basicNCAhead import BasicNCAHead


class ClassificationNCAHead(BasicNCAHead):
    def __init__(
        self,
        num_hidden_channels: int,
        num_classes: int,
        device: torch.device,
        avg_pool_size: int = 3,
        hidden_size: int = 64,
    ):
        super().__init__()
        self.num_hidden_channels = num_hidden_channels
        self.num_classes = num_classes
        self.device = device
        self.avg_pool_size = avg_pool_size
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.avg_pool_size, self.avg_pool_size)),
            nn.Flatten(1, 3),
            nn.Linear(
                num_hidden_channels * avg_pool_size**2,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes, bias=False),
        )
        self.classifier.to(device)

    def forward(self, x):
        return self.classifier(x)

    def freeze(self, freeze_last: bool = False):
        layers = [L for L in self.classifier.modules()]
        if not freeze_last:
            layers = layers[:-1]
        for layer in layers:
            layer.requires_grad_(False)
