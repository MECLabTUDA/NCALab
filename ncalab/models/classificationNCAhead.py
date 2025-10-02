from torch import nn

from .basicNCAhead import BasicNCAHead


class ClassificationNCAHead(BasicNCAHead):
    def __init__(
        self, num_hidden_channels, num_classes, device, avg_pool_size=3, hidden_size=64
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
        layers = self.classifier
        if not freeze_last:
            layers = self.classifier[:-1]
        for layer in layers:
            layer.requires_grad_(False)
