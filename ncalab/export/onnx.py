from os import PathLike

import torch


def export_onnx(nca, path: str | PathLike):
    dummy = torch.zeros((8, 16, 16, nca.num_channels)).to(nca.device)
    torch.onnx.export(nca, (dummy,), path, dynamo=True)
