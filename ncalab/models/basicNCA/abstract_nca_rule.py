import abc

import torch
from torch import nn


class AbstractNCARule(nn.Module, abc.ABC):
    def __init__(
        self, device: torch.device, input_size: int, hidden_size: int, output_size: int
    ):
        super().__init__()
        self.device: torch.device = device
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

    @abc.abstractmethod
    def freeze(self, freeze_last: bool = False) -> None:
        pass
