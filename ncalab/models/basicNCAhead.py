from torch import nn


class BasicNCAHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()

    def freeze(self, freeze_last: bool = True):
        raise NotImplementedError()
