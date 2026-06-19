from typing import Tuple


class Color:
    """
    Abstraction for RGBA color.

    4-float RGBA is used as an internal representation, but objects can be
    instantiated from different color formats.
    """

    def __init__(
        self,
        rgba4f: Tuple[float, float, float, float] | float,
        g: float = 0.0,
        b: float = 0.0,
        a: float = 1.0,
    ):
        if isinstance(rgba4f, tuple):
            self._rgba4f = rgba4f
        else:
            self._rgba4f = (rgba4f, g, b, a)

    @staticmethod
    def from_rgba4b(r: int, g: int, b: int, a: int = 255):
        assert (
            r >= 0
            and r <= 255
            and g >= 0
            and g <= 255
            and b >= 0
            and b <= 255
            and a >= 0
            and a <= 255
        )
        return Color((r / 255, g / 255, b / 255, a / 255))

    @property
    def rgba4f(self) -> Tuple[float, float, float, float]:
        return self._rgba4f

    @property
    def rgba4b(self) -> Tuple[int, int, int, int]:
        return (
            int(self.rgba4f[0] * 255.0),
            int(self.rgba4f[1] * 255.0),
            int(self.rgba4f[2] * 255.0),
            int(self.rgba4f[3] * 255.0),
        )

    @property
    def rgb3f(self) -> Tuple[float, float, float]:
        return self.rgba4f[:3]

    @staticmethod
    def from_hsva4f(h: float, s: float, v: float, a: float = 1.0):
        # TODO: implement
        raise NotImplementedError

    @property
    def hsva4f(self) -> Tuple[float, float, float, float]:
        # TODO: implement
        raise NotImplementedError
