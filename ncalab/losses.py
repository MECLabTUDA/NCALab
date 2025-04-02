import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]


class DiceScore(nn.Module):
    """ """

    def __init__(self):
        """"""
        super(DiceScore, self).__init__()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, smooth: float = 1
    ) -> torch.Tensor:
        """

        Args:
            input (_type_): _description_
            target (_type_): _description_
            smooth (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        x = torch.sigmoid(x)
        x = torch.flatten(x)
        y = torch.flatten(y)

        intersection = (x * y).sum()
        dice_score = (2.0 * intersection + smooth) / (x.sum() + y.sum() + smooth)
        return dice_score


class DiceBCELoss(nn.Module):
    """
    Combination of Dice and BCE Loss.
    """

    def __init__(self):
        """_summary_"""
        super(DiceBCELoss, self).__init__()
        self.dicescore = DiceScore()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, smooth: float = 1
    ) -> torch.Tensor:
        """_summary_

        Args:
            input (_type_): _description_
            target (_type_): _description_
            smooth (int, optional): _description_. Defaults to 1.

        Returns:
            torch.Tensor: Combination of Dice and BCE loss
        """
        x = torch.sigmoid(x)
        x = torch.flatten(x)
        y = torch.flatten(y)

        dice_loss = 1 - self.dicescore(x, y, smooth)
        BCE = F.binary_cross_entropy(x, y, reduction="mean")
        Dice_BCE = BCE + dice_loss
        return Dice_BCE
