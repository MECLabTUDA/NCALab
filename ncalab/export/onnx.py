from os import PathLike

import torch

from ..models import AbstractNCAModel


def export_onnx(nca: AbstractNCAModel, path: str | PathLike):
    dummy = torch.zeros((8, 16, 16, nca.num_channels)).to(nca.device)
    torch.onnx.export(nca, (dummy,), path, dynamo=True)
