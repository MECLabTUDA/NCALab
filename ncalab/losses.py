from typing import Literal

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]
from torch import nn  # type: ignore[import-untyped]


class DiceScore(nn.Module):
    """
    Pytorch Module that computes the Dice overlap score between two images.
    """

    def __init__(self, reduction: Literal["mean", "none", "sum"] = "mean"):
        super().__init__()
        self.reduction: Literal["mean", "none", "sum"] = reduction

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        """
        :param x: Reference Input
        :type x: torch.Tensor
        :param y: Other Input
        :type y: torch.Tensor
        :param smooth: Smooting factor, defaults to 1.0
        :type smooth: float

        :returns: Dice score
        :rtype: torch.Tensor
        """
        x = torch.sigmoid(x)
        intersection = x * y
        dice_score = (2.0 * intersection + smooth) / (x + y + smooth)
        if self.reduction == "none":
            return dice_score
        if self.reduction == "mean":
            return dice_score.mean()
        if self.reduction == "sum":
            return dice_score.sum()
        raise RuntimeError("Invalid reduction")


class DiceLoss(nn.Module):
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                f"Invalid reduction: {reduction}. "
                "Expected one of: 'none', 'mean', 'sum'."
            )
        self.dicescore: DiceScore = DiceScore()
        self.reduction: Literal["none", "mean", "sum"] = reduction

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        dice_loss: torch.Tensor = 1 - self.dicescore(x, y, smooth)

        if self.reduction == "none":
            return dice_loss
        if self.reduction == "mean":
            return dice_loss.mean()
        if self.reduction == "sum":
            return dice_loss.sum()
        raise RuntimeError("Invalid reduction")


class DiceBCELoss(nn.Module):
    """
    Combination of Dice and BCE Loss between two images.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.dicescore = DiceScore()
        self.reduction = reduction

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        """
        :param x: Reference Input
        :type x: torch.Tensor
        :param y: Other Input
        :type y: torch.Tensor
        :param smooth: Smooting factor, defaults to 1.0
        :type smooth: float

        :returns: Dice score
        :rtype: torch.Tensor
        """
        dice_loss = 1 - self.dicescore(x, y, smooth)
        x = torch.sigmoid(x)
        BCE = F.binary_cross_entropy(x, y, reduction=self.reduction)
        Dice_BCE = BCE + dice_loss
        return Dice_BCE


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.weight: torch.Tensor | None = weight
        self.gamma: float = gamma
        self.reduce = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        ce_loss = F.cross_entropy(
            x.reshape(-1, x.size(-1)),
            y.reshape(-1),
            weight=self.weight,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduce == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduce == "sum":
            focal_loss = focal_loss.sum()
        elif self.reduce != "none":
            raise ValueError(
                f"Invalid reduction: {self.reduce}. "
                + "Expected 'none', 'mean', or 'sum'."
            )
        return focal_loss
