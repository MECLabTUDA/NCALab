from os import PathLike

import torch


def export_onnx(nca, path: str | PathLike, optimize: bool = True):
    dummy = torch.zeros((8, 16, 16, nca.num_channels)).to(nca.device)
    onnx_program = torch.onnx.export(nca, dummy, dynamo=True)
    if optimize:
        onnx_program.optimize()
    onnx_program.save(path)
