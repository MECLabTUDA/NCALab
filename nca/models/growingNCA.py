import torch
import torch.nn.functional as F

from .basicNCA import BasicNCAModel


class GrowingNCAModel(BasicNCAModel):
    def __init__(
        self,
        device,
        num_image_channels: int,
        num_hidden_channels: int,
        fire_rate=0.5,
        hidden_size=128,
        use_alive_mask=False,
        learned_filters=0,
    ):
        super(GrowingNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            0,
            fire_rate,
            hidden_size,
            use_alive_mask,
            immutable_image_channels=False,
            learned_filters=learned_filters,
        )
        self.visualization_rows = ["input_image", "pred_image"]

    def loss(self, x, target):
        loss = F.mse(x[..., :num_image_channels].transpose(3, 1), target)
        return loss
