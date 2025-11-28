from typing import Dict

import segmentation_models_pytorch as smp  # type: ignore[import-untyped]
import torch  # type: ignore[import-untyped]

from ..losses import DiceBCELoss
from ..prediction import Prediction
from ..visualization import VisualBinaryImageSegmentation
from .basicNCA import BasicNCAModel


class SegmentationNCAModel(BasicNCAModel):
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
        pad_noise: bool = True,
        filter_padding: str = "reflect",
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

    def loss(self, pred: Prediction, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Dice + BCE loss.

        :param pred: Prediction.
        :param label: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        loss_segmentation_function = DiceBCELoss()
        loss_segmentation = loss_segmentation_function(
            pred.output_channels,
            label,
        )

        loss = loss_segmentation
        return {"total": loss}

    def metrics(self, pred: Prediction, label: torch.Tensor) -> Dict[str, float]:
        """
        Return dict of standard evaluation metrics.

        :param pred [torch.Tensor]: Predicted image.
        :param label [torch.Tensor]: Ground truth label.
        """
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred.output_channels.cpu().float(),
            label[:, None, :, :].cpu().long(),
            mode="binary",
            threshold=0.1,
        )
        tp = tp.squeeze().long()
        fp = fp.squeeze().long()
        fn = fn.squeeze().long()
        tn = tn.squeeze().long()
        iou_score = smp.metrics.iou_score(
            tp,
            fp,
            fn,
            tn,
            reduction="macro-imagewise",
        ).item()
        Dice = torch.mean(2.0 * (tp + 1.0) / (2.0 * tp + fp + fn + 1.0)).item()
        return {"IoU": iou_score, "Dice": Dice, "TP": tp, "FP": fp, "FN": fn, "TN": tn}
