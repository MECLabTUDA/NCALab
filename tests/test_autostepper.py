import pytest
import torch

from ncalab import AutoStepper


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
