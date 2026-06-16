from typing import Dict, Literal

import torch  # type: ignore[import-untyped]
import torchmetrics.segmentation

from ncalab.losses import DiceLoss
from ncalab.models.basicNCA import AbstractNCAModel
from ncalab.prediction import Prediction
from ncalab.visualization import VisualBinaryImageSegmentation


class BinarySegmentationNCAModel(AbstractNCAModel):
    """
    NCA model for binary pixel-wise image segmentation.

    Uses Dice score as the default validation metric.
    """

    def __init__(
        self,
        device: torch.device,
        num_image_channels: int = 3,
        num_hidden_channels: int = 16,
        fire_rate: float = 0.8,
        hidden_size: int = 128,
        num_learned_filters: int = 2,
        pad_noise: bool = False,
        filter_padding: Literal[
            "zero", "reflect", "replicate", "circular"
        ] = "circular",
        lambda_hidden: float = 1e-3,
        lambda_bce: float = 0.5,
        lambda_dice: float = 0.5,
        **kwargs,
    ):
        """
        :param device: Compute device.
        :type device: torch.device
        :param num_image_channels: Number of image channels, defaults to 3
        :type num_image_channels: int, optional
        :param num_hidden_channels: Number of hidden channels, defaults to 16
        :type num_hidden_channels: int, optional
        :param fire_rate: NCA fire rate, defaults to 0.8
        :type fire_rate: float, optional
        :param hidden_size: Number of neurons in hidden layer, defaults to 128
        :type hidden_size: int, optional
        :param num_learned_filters: Number of learned filters. If 0, use sobel, defaults to 2
        :type num_learned_filters: int, optional
        :param pad_noise: Whether to pad input images with noise, defaults to False
        :type pad_noise: bool, optional
        :param filter_padding: Padding type to use. Might affect reliance on spatial cues, defaults to "circular"
        :type filter_padding: Literal[ &quot;zero&quot;, &quot;reflect&quot;, &quot;replicate&quot;, &quot;circular&quot; ], optional
        :param lambda_hidden: Weight off hidden channel regularization, defaults to 1e-3
        :type lambda_hidden: float, optional
        :param lambda_bce: Weight of Binary Cross Entropy loss term, defaults to 0.5
        :type lambda_bce: float, optional
        :param lambda_dice: Weight of Dice loss term, defaults to 0.5
        :type lambda_dice: float, optional
        """
        super(BinarySegmentationNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            num_output_channels=1,
            plot_function=VisualBinaryImageSegmentation(),
            validation_metric="dice",
            fire_rate=fire_rate,
            hidden_size=hidden_size,
            use_alive_mask=False,
            immutable_image_channels=True,
            num_learned_filters=num_learned_filters,
            pad_noise=pad_noise,
            filter_padding=filter_padding,
            **kwargs,
        )
        self.metrics = {
            "dice": torchmetrics.segmentation.DiceScore(num_classes=2).to(self.device),
        }
        self.lambda_hidden = lambda_hidden
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.dice_loss = DiceLoss()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice

    def loss(self, pred: Prediction, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss as a weighted sum of Dice, BCE and hidden channel loss.

        :param pred: Prediction.
        :param label: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        y = pred.logits

        loss_dice = self.dice_loss(y, label)
        if label.ndim == 3:
            label = label.unsqueeze(1)
        loss_bce = self.bce_loss(y, label.float())
        loss_hidden = torch.mean(torch.abs(pred.hidden_channels))

        loss = (
            self.lambda_dice * loss_dice
            + self.lambda_bce * loss_bce
            + self.lambda_hidden * loss_hidden
        )
        return {
            "total": loss,
            "dice": loss_dice.detach(),
            "bce": loss_bce.detach(),
            "hidden": loss_hidden.detach(),
        }

    def post_prediction(self, prediction: Prediction) -> Prediction:
        prediction.mask = torch.sigmoid(prediction.logits) > 0.5
        return prediction
