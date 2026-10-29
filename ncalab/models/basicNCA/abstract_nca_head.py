import abc

import torch
from torch import nn


class AbstractNCAHead(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self.optimizer: torch.optim.Optimizer | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor
        :type x: torch.Tensor

        :returns: NotImplemented, subclasses are required to implement this method.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def freeze(self, freeze_last: bool = True) -> None:
        """
        Freeze head weights.

        :param freeze_last: Whether to freeze the last layer (if applicable), defaults to True
        :type freeze_last: bool, optional
        :returns: NotImplemented, subclasses are required to implement this method.
        """
        raise NotImplementedError
