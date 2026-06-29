from typing import Dict

import torch
import pytest

from ncalab import (
    AbstractNCAModel,
    Animator,
    Color,
    Prediction,
    get_compute_device,
    string_ellipsis,
)


def test_string_ellipsis():
    assert string_ellipsis("hello world", max_len=100) == "hello world"
    assert string_ellipsis("hello world", max_len=(len("hello "))) == "HW"
    assert string_ellipsis("helloworld", max_len=3) == "he…"


def test_color():
    rgba4f = Color((0.1, 0.3, 0.3, 0.7))
    assert rgba4f.rgba4b == (
        int(0.1 * 255),
        int(0.3 * 255),
        int(0.3 * 255),
        int(0.7 * 255),
    )


class DummyModel(AbstractNCAModel):
    def __init__(self, device):
        super().__init__(device, 3, 12, 5)

    def loss(self, pred: Prediction, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}


def test_animator():
    device = get_compute_device("cpu")
    model = DummyModel(device)
    try:
        animator = Animator(  # noqa
            model, torch.randn(8, model.num_channels, 32, 32), hidden=True, overlay=True
        )
    except Exception as e:
        pytest.fail(str(e))
