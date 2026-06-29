import pytest
import torch
import torch.nn.functional as F

from ncalab import DiceBCELoss, DiceLoss, DiceScore, FocalLoss


@pytest.fixture
def logits():
    return torch.tensor([0.0, 2.0, -2.0, 0.5])


@pytest.fixture
def target():
    return torch.tensor([0.0, 1.0, 0.0, 1.0])


def test_dicescore_returns_scalar_tensor(logits, target):
    score = DiceScore()(logits, target)
    assert isinstance(score, torch.Tensor)
    assert score.ndim == 0


def test_dicescore_matches_manual_computation(logits, target):
    score = DiceScore()(logits, target)
    x = torch.sigmoid(logits).flatten()
    y = target.flatten()
    expected = (2 * (x * y).sum() + 1.0) / (x.sum() + y.sum() + 1.0)
    assert torch.allclose(score, expected)


def test_dicescore_flatten_inputs():
    logits = torch.tensor([[0.0, 2.0], [-2.0, 0.5]])
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    score = DiceScore()(logits, target)
    score_flat = DiceScore()(logits.flatten(), target.flatten())
    assert torch.allclose(score, score_flat)


def test_score_valid_range(logits, target):
    score = DiceScore()(logits, target)
    assert 0 <= score.item() <= 1


def test_dice_loss_score(logits, target):
    dice = DiceScore()(logits, target)
    loss = DiceLoss()(logits, target)
    assert torch.allclose(loss, 1 - dice)


def test_dicebce_matches_manual_formula(logits, target):
    loss = DiceBCELoss()(logits, target)
    x = torch.sigmoid(logits)
    bce = F.binary_cross_entropy(x, target)
    dice = DiceScore()(logits, target)
    expected = bce + (1 - dice)
    assert torch.allclose(loss, expected)


def test_focal_forward_without_weights():
    loss_fn = FocalLoss()
    inputs = torch.tensor(
        [
            [[2.0, 0.5], [0.5, 2.0]],
            [[0.2, 1.8], [2.5, 0.3]],
        ],
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    loss = loss_fn(inputs, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_focal_forward_with_weights():
    weights = torch.tensor([1.0, 2.0])
    loss_fn = FocalLoss(weight=weights)
    inputs = torch.tensor(
        [
            [[2.0, 0.5], [0.5, 2.0]],
            [[0.2, 1.8], [2.5, 0.3]],
        ]
    )
    targets = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    loss = loss_fn(inputs, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0
