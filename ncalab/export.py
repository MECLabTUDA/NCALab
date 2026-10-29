from functools import wraps

import numpy as np


def c_header_exporter(func):
    @wraps(func)
    def wrapper(*args, indent: int = 0, **kwargs):
        if not isinstance(indent, int):
            raise TypeError("Indentation must be an int")

        result = func(*args, **kwargs)
        result = "\n".join([" " * indent * 4 + line for line in result.splitlines()])

    return wrapper


class HeaderExportable:
    def __init__(self):
        pass

    def to_c_header(self) -> str:
        raise NotImplementedError


@c_header_exporter
def int_to_c_header(x: int, hex: bool = False):
    if hex:
        return f"{x:x}"
    return str(x)


@c_header_exporter
def float_to_c_header(x: float, precision: int | None = None) -> str:
    if precision is None:
        return str(x)
    assert precision >= 0
    if precision == 0:
        return str(int(x))
    return f"{x:.{precision}}"


@c_header_exporter
def str_to_c_header(x: str) -> str:
    return f'"{x}"'


@c_header_exporter
def array_to_c_header(x: np.ndarray, precision: int | None = None) -> str:
    xf = x.flatten()
    xs = [float_to_c_header(f, precision) for f in xf]
    line = "{ " + ",".join(xs) + " }"
    return line


@c_header_exporter
def ndarray_to_c_header(x: np.ndarray) -> str:
    xc = np.ascontiguousarray(x)  # noqa
    # TODO
    return ""


@c_header_exporter
def decl_to_c_header(t: str, ident: str, value: object | None = None):
    if value is None:
        return f"{t} {ident};"
    return f"{t} {ident} = {value};"


@c_header_exporter
def arrdecl_to_c_header(
    t: str, ident: str, dims: list[str], arr: np.ndarray | None = None
):
    if arr is not None:
        assert len(dims) == len(arr)
    s = f"{t} {ident}"
    for dim in dims:
        s += f"[{dim}]"
    if arr is not None:
        s += " = "
        s += ndarray_to_c_header(arr)
    s += ";"
    return s
