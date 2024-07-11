"""Exporting NCA weights to C headers.
"""

import torch
from pathlib import PosixPath, Path


def export_header(
    infile: str | Path | PosixPath,
    outfile: str | Path | PosixPath,
    reduce_arrays: bool = True,
    number_type: str = "float",
    imports: list | None = None,
):
    ## prepare preamble
    # guard
    preamble = "#pragma once\n"
    # add imports if any
    if imports:
        for header in imports:
            preamble += f'#include "{header}"'
    preamble += "\n\n"

    weights = torch.load(infile)

    with open(outfile, "w") as f:
        f.write(preamble)
        for i, key in enumerate(weights):
            name = key.replace(".", "_")
            arr = weights[key]
            if len(arr.shape) == 2:
                f.write(f"{number_type} {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n")
                for i in range(arr.shape[0]):
                    f.write("    {")
                    W = [f"{w:.8f}" for w in arr[i]]
                    f.write(f"{', '.join(W)}")
                    f.write("}")
                    if i != arr.shape[0]:
                        f.write(",")
                    f.write("\n")
                f.write("};\n")
            elif len(arr.shape) == 1:
                f.write(f"{number_type} {name}[{arr.shape[0]}] = {{\n")
                W = [f"{w:.8f}" for w in arr]
                f.write(f"    {', '.join(W)}\n")
                f.write("};\n")
