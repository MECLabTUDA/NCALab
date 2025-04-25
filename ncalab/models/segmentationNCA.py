from .basicNCA import BasicNCAModel
from .splitNCA import SplitNCAModel
from ..losses import DiceBCELoss
from ..visualization import show_batch_binary_segmentation
from ..utils import pad_input

from typing import Dict, Tuple

import torch  # type: ignore[import-untyped]
import segmentation_models_pytorch as smp  # type: ignore[import-untyped]


class SegmentationNCAModel(BasicNCAModel):
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
    ):
        """
        Instantiate an image segmentation model based on NCA.

        Args:
            device (torch.device): Compute device
            num_image_channels (int): _description_
            num_hidden_channels (int): _description_
            num_classes (int): _description_. Defaults to 1.
            fire_rate (float, optional): _description_. Defaults to 0.8.
            hidden_size (int, optional): _description_. Defaults to 128.
            learned_filters (int, optional): _description_. Defaults to 2.
            pad_noise (bool, optional): _description_. Defaults to True.
        """
        self.num_classes = num_classes
        super(SegmentationNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            num_output_channels=num_classes,
            fire_rate=fire_rate,
            hidden_size=hidden_size,
            use_alive_mask=False,
            immutable_image_channels=True,
            num_learned_filters=num_learned_filters,
            pad_noise=pad_noise,
        )
        self.plot_function = show_batch_binary_segmentation
        self.validation_metric = "Dice"

    def segment(self, image, return_all=False, return_steps=False, **kwargs):
        if return_all:
            return_steps = True
        with torch.no_grad():
            x = image.clone()
            x = pad_input(x, self, noise=self.pad_noise)
            x = x.permute(0, 2, 3, 1)
            x = self(x, **kwargs, return_steps=return_steps)
            if return_steps:
                steps = 0
                x, steps = x

            class_channels = x[
                ..., self.num_image_channels + self.num_hidden_channels :
            ]

            if return_all:
                return class_channels, x, steps
            if return_steps:
                return x, steps
            return class_channels

    def loss(self, x, y):
        class_channels = x[..., self.num_image_channels + self.num_hidden_channels :]

        loss_segmentation_function = DiceBCELoss()
        loss_segmentation = loss_segmentation_function(
            class_channels.permute(0, 3, 1, 2),
            y,
        )

        loss = loss_segmentation
        return {"total": loss}

    def metrics(
        self,
        image,
        label,
        steps: int,
    ):
        x_pred = self(image, steps=steps)
        outputs = x_pred[
            ..., self.num_image_channels + self.num_hidden_channels :
        ].permute(0, 3, 1, 2)
        tp, fp, fn, tn = smp.metrics.get_stats(
            outputs.cpu(),
            label[:, None, :, :].cpu().long(),
            mode="binary",
            threshold=0.1,
        )
        tp = tp.squeeze()
        fp = fp.squeeze()
        fn = fn.squeeze()
        tn = tn.squeeze()
        iou_score = smp.metrics.iou_score(
            tp,
            fp,
            fn,
            tn,
            reduction="macro-imagewise",
        ).item()
        Dice = torch.mean(2.0 * (tp + 1.0) / (2.0 * tp + fp + fn + 1.0)).item()
        return {
            "IoU": iou_score,
            "Dice": Dice,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "prediction": x_pred,
        }

    def validate(
        self,
        image,
        label,
        steps: int,
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        metrics = self.metrics(image, label, steps)
        return metrics, metrics["prediction"]
