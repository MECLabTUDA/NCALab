from typing import Dict

import pytest
import torch

from ncalab import (
    AbstractNCAModel,
    AutoStepper,
    Prediction,
    get_compute_device,
)


def test_constructor_assertions():
    with pytest.raises(AssertionError):
        AutoStepper(min_steps=0)

    with pytest.raises(AssertionError):
        AutoStepper(plateau=0)

    with pytest.raises(AssertionError):
        AutoStepper(min_steps=10, max_steps=5)

    with pytest.raises(AssertionError):
        AutoStepper(min_steps=10, max_steps=10)


def test_score():
    stepper = AutoStepper(verbose=False)
    stepper.cooldown = stepper.plateau

    stepper.hidden_i = torch.ones((8, 10, 24, 24))
    stepper.hidden_i_1 = torch.zeros((8, 10, 24, 24))
    score = stepper._score()
    assert torch.isclose(score, torch.tensor(1.0))
    for i in range(100):
        assert not stepper._check(stepper.min_steps)

    stepper.hidden_i = torch.zeros((8, 10, 24, 24))
    stepper.hidden_i_1 = torch.zeros((8, 10, 24, 24))
    score = stepper._score()
    assert torch.isclose(score, torch.tensor(0.0))
    for i in range(stepper.plateau - 1):
        assert not stepper._check(stepper.min_steps + i)
    assert stepper._check(stepper.min_steps + stepper.plateau)


class DummyModel(AbstractNCAModel):
    def __init__(self, device):
        super().__init__(device, 3, 12, 5)

    def loss(self, pred: Prediction, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}


def test_run():
    device = get_compute_device("cpu")
    nca = DummyModel(device)
    stepper = AutoStepper(verbose=False)
    try:
        stepper(nca, torch.randn(8, nca.num_channels, 32, 32))
    except Exception as e:
        pytest.fail(str(e))
