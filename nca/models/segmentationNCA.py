import torch
import torch.nn.functional as F

from .basicNCA import BasicNCAModel
from ..losses import DiceBCELoss, DiceScore
from ..visualization import show_batch_binary_segmentation


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
        lambda_activity: float = 0.005,
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

    def segment(self, image, steps=100):
        with torch.no_grad():
            x = image.clone()
            x = self(x, steps=steps)
            hidden_channels = x[
                ...,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
            ]

            class_channels = x[
                ..., self.num_image_channels + self.num_hidden_channels :
            ]
            return class_channels

    def loss(self, x, y):
        hidden_channels = x[..., self.num_image_channels : -self.num_output_channels]
        class_channels = x[..., self.num_image_channels + self.num_hidden_channels :]

        loss_segmentation_function = DiceBCELoss()
        loss_segmentation = loss_segmentation_function(
            class_channels.transpose(3, 1), y  # B, W, H, C --> B, C, W, H
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
        x: torch.Tensor,
        target: torch.Tensor,
        steps: int,
        batch_iteration: int,
        summary_writer=None,
    ):
        y_pred = self.segment(x, steps)
        metric = DiceScore()
        dice = metric(y_pred, target)
        if summary_writer:
            summary_writer.add_scalar(
                "Acc/val_dice_score", dice, batch_iteration
            )
        return dice