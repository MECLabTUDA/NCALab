import torch
import torch.nn.functional as F

import numpy as np

from .basicNCA import BasicNCAModel
from ..visualization import show_batch_growing


class GrowingNCAModel(BasicNCAModel):
    def __init__(
        self,
        device,
        num_image_channels: int,
        num_hidden_channels: int,
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        learned_filters: int = 2,
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
            num_learned_filters=learned_filters,
        )
        self.plot_function = show_batch_growing

    def loss(self, x, target):
        loss = F.mse_loss(x[..., : self.num_image_channels], target)
        return loss

    def validate(
        self,
        *args,
        **kwargs
    ):
        pass

    def grow(self, width, height, steps: int = 100) -> np.ndarray:
        seed = torch.zeros((1, self.num_channels, width, height)).to(self.device)
        seed[:, 3:, :, :] = 1.0
        out = self.forward(seed.permute(0, 2, 3, 1), steps=steps)
        out = out[..., :3].detach().cpu().numpy()[0]
        out = np.clip(out, 0, 1)
        return out
