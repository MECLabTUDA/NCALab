import torch
from torch import nn


class DiceBCELoss(nn.Module):
    r"""Dice BCE Loss

    Addition of Dice loss and Binary Cross Entropy, as described in MedNCA."""

    def __init__(self, useSigmoid=True):
        r"""Initialisation method of DiceBCELoss
        #Args:
            useSigmoid: Whether to use sigmoid
        """
        self.useSigmoid = useSigmoid
        super(DiceBCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        r"""Forward function
        #Args:
            input: input array
            target: target array
            smooth: Smoothing value
        """
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)

        intersection = (input * target).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            input.sum() + target.sum() + smooth
        )
        BCE = torch.nn.functional.binary_cross_entropy(input, target, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
