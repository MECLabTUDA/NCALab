import torch

from .basicNCA import BasicNCAModel
from ..losses import DiceBCELoss
from ..visualization import show_batch_binary_segmentation
from ..utils import pad_input

import segmentation_models_pytorch as smp  # type: ignore[import-untyped]


class SegmentationNCAModel(BasicNCAModel):
    def __init__(
        self,
        device,
        num_image_channels: int,
        num_hidden_channels: int,
        num_classes: int,
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        immutable_image_channels: bool = True,
        learned_filters: int = 2,
        lambda_activity: float = 0.0,
    ):
        """_summary_

        Args:
            device (_type_): _description_
            num_image_channels (int): _description_
            num_hidden_channels (int): _description_
            num_classes (int): _description_
            fire_rate (float, optional): _description_. Defaults to 0.5.
            hidden_size (int, optional): _description_. Defaults to 128.
            use_alive_mask (bool, optional): _description_. Defaults to False.
            immutable_image_channels (bool, optional): _description_. Defaults to True.
            learned_filters (int, optional): _description_. Defaults to 2.
        """
        self.num_classes = num_classes
        self.lambda_activity = lambda_activity
        super(SegmentationNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            num_classes,
            fire_rate,
            hidden_size,
            use_alive_mask,
            immutable_image_channels,
            learned_filters,
        )
        self.plot_function = show_batch_binary_segmentation

    def segment(
        self, image, return_all=False, pad_noise=False, return_steps=False, **kwargs
    ):
        if return_all:
            return_steps = True
        with torch.no_grad():
            x = image.clone()
            x = pad_input(x, self, noise=pad_noise)
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
        hidden_channels = x[..., self.num_image_channels : -self.num_output_channels]
        class_channels = x[..., self.num_image_channels + self.num_hidden_channels :]

        loss_segmentation_function = DiceBCELoss()
        loss_segmentation = loss_segmentation_function(
            class_channels.permute(0, 3, 1, 2),
            y,
        )

        loss_activity = torch.sum(torch.square(hidden_channels)) / (
            x.shape[0] * x.shape[1] * x.shape[2]
        )

        loss = (
            1 - self.lambda_activity
        ) * loss_segmentation + self.lambda_activity * loss_activity
        return loss

    def validate(
        self,
        dataloader_val,
        steps: int,
        batch_iteration: int,
        summary_writer=None,
        pad_noise: bool = False,
    ):
        self.eval()
        TP = []
        FP = []
        FN = []
        TN = []

        dice = smp.losses.DiceLoss("binary", from_logits=False)
        dice_scores = []

        with torch.no_grad():
            for images, labels in dataloader_val:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.segment(
                    images, steps=steps, pad_noise=pad_noise
                ).permute(0, 3, 1, 2)
                dice_score = 1 - dice(labels, outputs)
                dice_scores.append(dice_score.item())
                tp, fp, fn, tn = smp.metrics.get_stats(
                    outputs, labels[:, None, :, :].long(), mode="binary", threshold=0.5
                )
                TP.append(tp[:, 0])
                FP.append(fp[:, 0])
                FN.append(fn[:, 0])
                TN.append(tn[:, 0])
        f1_score = smp.metrics.f1_score(
            torch.cat(TP),
            torch.cat(FP),
            torch.cat(FN),
            torch.cat(TN),
            reduction="micro",
        )
        if summary_writer:
            summary_writer.add_scalar("Acc/val_F1", f1_score.item(), batch_iteration)
        return f1_score.item()
