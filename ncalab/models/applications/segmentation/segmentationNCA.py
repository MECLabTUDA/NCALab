from typing import Dict, Literal

import torch  # type: ignore[import-untyped]
import torchmetrics.segmentation

from ncalab.losses import DiceLoss
from ncalab.models.basicNCA import AbstractNCAModel
from ncalab.prediction import Prediction
from ncalab.visualization import VisualBinaryImageSegmentation


class SegmentationNCAModel(AbstractNCAModel):
    """
    Model used for image segmentation.

    Uses Dice score as the default validation metric. Currently, only binary segmentation
    masks are supported.
    """

    def __init__(
        self,
        device: torch.device,
        num_image_channels: int = 3,
        num_hidden_channels: int = 16,
        num_classes: int = 1,
        fire_rate: float = 0.8,
        hidden_size: int = 128,
        num_learned_filters: int = 2,
        pad_noise: bool = False,
        filter_padding: Literal[
            "zero", "reflect", "replicate", "circular"
        ] = "circular",
        lambda_hidden: float = 1e-3,
        **kwargs,
    ):
        """
        :param device [torch.device]: Compute device.
        :param num_image_channels [int]: Number of image channels. Defaults to 3.
        :param num_hidden_channels [int]: Number of hidden channels. Defaults to 16.
        :param num_classes [int]: Number of classes. Defaults to 1.
        :param fire_rate [float]: NCA fire rate. Defaults to 0.8.
        :param hidden_size [int]: Number of neurons in hidden layer. Defaults to 128.
        :param learned_filters [int]: Number of learned filters. If 0, use sobel. Defaults to 2.
        :param pad_noise [bool]: Whether to pad input images with noise. Defaults to True.
        :param filter_padding [str]: Padding type to use. Might affect reliance on spatial cues. Defaults to "circular".
        """
        self.num_classes = num_classes
        super(SegmentationNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            num_output_channels=num_classes,
            plot_function=VisualBinaryImageSegmentation(),
            validation_metric="Dice",
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
            # FIXME: set number of classes according to parameter
            "Dice": torchmetrics.segmentation.DiceScore(num_classes=2).to(self.device),
        }
        self.lambda_hidden = lambda_hidden
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.dice_loss = DiceLoss()

    def loss(self, pred: Prediction, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Dice loss.

        :param pred: Prediction.
        :param label: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        y = pred.output_channels
        loss_dice = self.dice_loss(
            y,
            label,
        )
        loss_bce = self.bce_loss(y.flatten(), label.flatten())
        loss_hidden = torch.mean(torch.abs(pred.hidden_channels))

        loss = loss_dice + 0.1 * loss_bce + self.lambda_hidden * loss_hidden
        return {
            "total": loss,
            "dice": loss_dice,
            "bce": loss_bce,
            "hidden": loss_hidden,
        }

    def _post_forward_step(self, x: torch.Tensor) -> torch.Tensor:
        x[:, -self.num_output_channels :, :, :] = torch.sigmoid(
            x[:, -self.num_output_channels :, :, :]
        )
        return x

    def post_prediction(self, prediction: Prediction) -> Prediction:
        logits = (prediction.logits > 0.5).long().squeeze(1)
        return Prediction(
            prediction.model,
            prediction.steps,
            prediction.output_image,
            logits,
            prediction.head_prediction,
        )
