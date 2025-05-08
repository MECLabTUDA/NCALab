import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]


class DiceScore(nn.Module):
    """
    Pytorch Module that computes the Dice overlap score between two images.
    """

    def __init__(self):
        """
        Constructor.
        """
        super(DiceScore, self).__init__()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        """
        :param x [torch.Tensor]: Reference Input
        :param y [torch.Tensor]: Other Input

        :returns [torch.Tensor]: Dice score as Tensor
        """
        x = torch.sigmoid(x)
        x = torch.flatten(x)
        y = torch.flatten(y)

        intersection = (x * y).sum()
        dice_score = (2.0 * intersection + smooth) / (x.sum() + y.sum() + smooth)
        return dice_score


class DiceBCELoss(nn.Module):
    """
    Combination of Dice and BCE Loss between two images.
    """

    def __init__(self):
        """
        Constructor.
        """
        super(DiceBCELoss, self).__init__()
        self.dicescore = DiceScore()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        """
        :param x [torch.Tensor]: Reference Input.
        :param y [torch.Tensor]: Other Input.
        :param smooth [float]: Smoothing factor. Defaults to 1.0

        :returns: Combination of Dice and BCE loss
        """
        x = torch.sigmoid(x)
        x = torch.flatten(x)
        y = torch.flatten(y)

        dice_loss = 1 - self.dicescore(x, y, smooth)
        BCE = F.binary_cross_entropy(x, y, reduction="mean")
        Dice_BCE = BCE + dice_loss
        return Dice_BCE
