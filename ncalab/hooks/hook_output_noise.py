import torch

from .hook import Hook

class OutputNoiseHook(Hook):
    def __init__(self, dx_noise: float):
        self.dx_noise = dx_noise

    def pre_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def post_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def pre_perceive(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def pre_update(self, x: torch.Tensor) -> torch.Tensor:
        return x
