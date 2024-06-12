import torch.nn.functional as F

from .basicNCA import BasicNCAModel
from ..visualization import show_batch_growing


class GrowingNCAModel(BasicNCAModel):
    def __init__(
        self,
        device,
        num_image_channels: int,
        num_hidden_channels: int,
        fire_rate=0.5,
        hidden_size=128,
        use_alive_mask=False,
        learned_filters=2,
    ):
        """_summary_

        Args:
            device (_type_): _description_
            num_image_channels (int): _description_
            num_hidden_channels (int): _description_
            fire_rate (float, optional): _description_. Defaults to 0.5.
            hidden_size (int, optional): _description_. Defaults to 128.
            use_alive_mask (bool, optional): _description_. Defaults to False.
            learned_filters (int, optional): _description_. Defaults to 2.
        """
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
        self.plot_function = show_batch_growing

    def loss(self, x, target):
        loss = F.mse_loss(x[..., : self.num_image_channels], target)
        return loss
