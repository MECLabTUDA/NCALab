import abc
from typing import override

import numpy as np
import torch


class TemporalEncoding(abc.ABC):
    def __init__(self):
        self.normalization: float = 1

    @abc.abstractmethod
    def encode(self, timestep: int) -> list[float]:
        raise NotImplementedError

    def dim(self):
        return len(self.encode(0))

    def vector(self, timestep: int, x: torch.Tensor):
        enc = self.encode(timestep)
        return torch.mul(
            torch.ones((x.shape[0], len(enc), x.shape[2], x.shape[3])),
            torch.tensor(enc, dtype=x.dtype).view(1, len(enc), 1, 1),
        )


class LinearTemporalEncoding(TemporalEncoding):
    def __init__(self):
        super().__init__()

    @override
    def encode(self, timestep: int) -> list[float]:
        return [timestep / self.normalization]


class SinusoidalTemporalEncoding(TemporalEncoding):
    def __init__(self):
        super().__init__()

    @override
    def encode(self, timestep: int):
        return [
            np.sin(2 * np.pi * timestep / self.normalization),
            np.cos(2 * np.pi * timestep / self.normalization),
        ]


temporal_encodings = {
    "none": None,
    "linear": LinearTemporalEncoding,
    "sinusoidal": SinusoidalTemporalEncoding,
}
