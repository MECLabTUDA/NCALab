import torch


class Hook:
    def __init__(self, *args, **kwargs):
        pass

    def pre_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def post_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def pre_perceive(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def pre_update(self, x: torch.Tensor) -> torch.Tensor:
        return x
