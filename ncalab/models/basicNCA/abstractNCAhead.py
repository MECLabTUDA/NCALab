import abc

import torch
from torch import nn


class AbstractNCAHead(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor
        :type x: torch.Tensor

        :returns: NotImplemented, subclasses are required to implement this method.
        """
        return NotImplemented

    @abc.abstractmethod
    def freeze(self, freeze_last: bool = True) -> None:
        """
        Freeze head weights.

        :param freeze_last: Whether to freeze the last layer (if applicable), defaults to True
        :type freeze_last: bool, optional
        :returns: NotImplemented, subclasses are required to implement this method.
        """
        # FIXME: properly annotate this method, see https://github.com/python/mypy/issues/363
        return NotImplemented  # type: ignore
