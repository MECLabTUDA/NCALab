import torch
from torch import nn


class DiceBCELoss(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        """_summary_

        Args:
            useSigmoid (bool, optional): _description_. Defaults to True.
        """
        super(DiceBCELoss, self).__init__()

    def forward(self, x, target, smooth=1):
        """_summary_

        Args:
            input (_type_): _description_
            target (_type_): _description_
            smooth (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        x = torch.sigmoid(x)
        x = torch.flatten(x)
        target = torch.flatten(target)

        intersection = (x * target).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            x.sum() + target.sum() + smooth
        )
        BCE = torch.nn.functional.binary_cross_entropy(x, target, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
