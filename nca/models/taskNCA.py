import torch
import torch.nn.functional as F

from .basicNCA import BasicNCAModel


class TaskNCAModel(BasicNCAModel):
    def __init__(
        self,
        device,
        num_image_channels: int,
        num_hidden_channels: int,
        num_output_channels: int,
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        immutable_image_channels: bool = True,
        learned_filters: int = 0,
    ):
        """_summary_

        Args:
            device (Pytorch device descriptor): _description_
            num_image_channels (int): _description_
            num_hidden_channels (int): _description_
            num_classes (int): _description_
            fire_rate (float, optional): _description_. Defaults to 0.5.
            hidden_size (int, optional): _description_. Defaults to 128.
            use_alive_mask (bool, optional): _description_. Defaults to False.
            immutable_image_channels (bool, optional): _description_. Defaults to True.
            learned_filters (int, optional): _description_. Defaults to 2.
            lambda_activity (float, optional): Activity loss weight, penalizing high NCA activity. Defaults to 0.01.
            pixel_wise_loss (bool, optional): Whether a prediction per pixel is desired, like in self classifying MNIST. Defaults to False.
        """
        super(TaskNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            num_output_channels,
            fire_rate,
            hidden_size,
            use_alive_mask,
            immutable_image_channels,
            learned_filters,
        )

    def forward(self, x: torch.Tensor, steps: int = 1):
        x = super().forward(x, steps)
        return x

    def validate(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        steps: int,
        batch_iteration: int,
        summary_writer=None,
    ):
        pass

    def loss(self, x, target):
        hidden_channels = x[..., self.num_image_channels : -self.num_output_channels]
        class_channels = x[..., self.num_image_channels + self.num_hidden_channels :]

        y_pred = F.softmax(class_channels, dim=-1)
        loss_mse = F.mse_loss(
            y_pred.float(),
            target.float(),
        )

        loss = loss_mse
        return loss