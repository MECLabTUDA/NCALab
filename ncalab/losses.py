import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]
from torch import nn  # type: ignore[import-untyped]


class DiceScore(nn.Module):
    """
    Pytorch Module that computes the Dice overlap score between two images.
    """

    def __init__(self):
        super().__init__()

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
        return dice_score


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dicescore = DiceScore()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, smooth: float = 1.0
    ) -> torch.Tensor:
        dice_loss = 1 - self.dicescore(x, y, smooth)
        return dice_loss


class DiceBCELoss(nn.Module):
    """
    Combination of Dice and BCE Loss between two images.
    """

    def __init__(self, reduction="none"):
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


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, device="cpu", reduction="none"):
        super().__init__(weight)
        self.gamma = gamma
        self.weight = weight
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.reduce = reduction

    def forward(self, _input, _target):
        focal_loss = 0
        for i in range(len(_input)):
            ce_loss = self.ce_loss(
                _input[i].view(-1, _input[i].size()[-1]), _target[i].view(-1)
            )
            pt = torch.exp(-ce_loss)
            cur_focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            focal_loss = focal_loss + cur_focal_loss
        if self.weight is not None:
            focal_loss = focal_loss / self.weight.sum()
            return focal_loss.to(self.device)
        focal_loss = focal_loss / torch.tensor(len(_input))
        return focal_loss.to(self.device)
