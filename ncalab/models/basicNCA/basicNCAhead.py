import torch
from torch import nn


class BasicNCAHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        _summary_

        :param x: _description_
        :type x: _type_
        :raises NotImplementedError: Subclasses are required to implement this method.
        """
        raise NotImplementedError()

    def freeze(self, freeze_last: bool = True):
        """
        _summary_

        :param freeze_last: _description_, defaults to True
        :type freeze_last: bool, optional
        :raises NotImplementedError: Subclasses are required to implement this method.
        """
        raise NotImplementedError()
