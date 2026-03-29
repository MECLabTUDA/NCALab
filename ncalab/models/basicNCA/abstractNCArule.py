import abc

import torch
from torch import nn


class AbstractNCARule(nn.Module, abc.ABC):
    def __init__(
        self, device: torch.device, input_size: int, latent_size: int, output_size: int
    ):
        super(AbstractNCARule, self).__init__()
        self.device = device
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size

    @abc.abstractmethod
    def freeze(self, freeze_last: bool = False) -> None:
        pass
