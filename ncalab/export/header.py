from __future__ import annotations
import os
from pathlib import PosixPath, Path

import torch

from ..models.basicNCA import BasicNCAModel


def tensor_to_C(tensor: torch.Tensor, name: str, number_type: str = "float"):
    s = ""
    if len(tensor.shape) == 2:
        s += f"{number_type} {name}[{tensor.shape[0]}][{tensor.shape[1]}] = {{\n"
        for i in range(tensor.shape[0]):
            s += "    {"
            W = [f"{w:.8f}" for w in tensor[i]]
            s += f"{', '.join(W)}"
            s += "}"
            if i != tensor.shape[0]:
                s += ","
            s += "\n"
        s += "};\n"
    elif len(tensor.shape) == 1:
        s += f"{number_type} {name}[{tensor.shape[0]}] = {{\n"
        W = [f"{w:.8f}" for w in tensor]
        s += f"    {', '.join(W)}\n"
        s += "};\n"
    return s


def export_header(
    nca: BasicNCAModel,
    outfile: str | Path | PosixPath,
    number_type: str = "float",
    imports: list | None = None,
):
    """
    Export parameters of an NCA model to a C header.

    :param nca: _description_
    :type nca: BasicNCAModel
    :param outfile: _description_
    :type outfile: str | Path | PosixPath
    :param number_type: _description_, defaults to "float"
    :type number_type: str, optional
    :param imports: _description_, defaults to None
    :type imports: list | None, optional
    """
    ## prepare preamble
    # guard
    preamble = f"#pragma once{os.linesep}"
    # add imports if any
    if imports:
        for header in imports:
            preamble += f'#include "{header}"{os.linesep}'
    preamble += f"{os.linesep}{os.linesep}"

    with open(outfile, "w") as f:
        f.write(preamble)
        W_0 = nca.rule.network[0].weight.data
        b_0 = nca.rule.network[0].bias.data
        W_1 = nca.rule.network[1].weight.data
        assert type(W_0) is torch.Tensor
        assert type(b_0) is torch.Tensor
        assert type(W_1) is torch.Tensor
        f.write(tensor_to_C(W_0, "weight_0", number_type))
        f.write(tensor_to_C(b_0, "bias_0", number_type))
        f.write(tensor_to_C(W_1, "weight_1", number_type))
