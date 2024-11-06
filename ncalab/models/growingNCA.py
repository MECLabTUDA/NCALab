import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

import numpy as np

from .basicNCA import BasicNCAModel
from ..visualization import show_batch_growing


class GrowingNCAModel(BasicNCAModel):
    def __init__(
        self,
        device: torch.device,
        num_image_channels: int,
        num_hidden_channels: int,
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        **kwargs,
    ):
        """NCA Model class for "growing" tasks.

        This specialization of the BasicNCAModel has some interesting properties.
        For instance, it has no output channels, as the growing task directly
        manipulates the input image channels.

        Args:
            device (device): Pytorch device descriptor.
            num_image_channels (int): Number of channels reserved for input image.
            num_hidden_channels (int): Number of hidden channels (communication channels).
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
            **kwargs,
        )
        self.plot_function = show_batch_growing

    def loss(self, x, y):
        """Implements a simple MSE loss between target and prediction.

        Args:
            x (Tensor): Prediction
            y (Tensor): Target

        Returns:
            Tensor: MSE Loss
        """
        loss = F.mse_loss(x[..., : self.num_image_channels], y)
        return loss

    def validate(self, *args, **kwargs):
        """We typically don't validate during training of Growing NCA."""
        pass

    def grow(
        self, width: int, height: int, steps: int = 100, save_steps=False
    ) -> np.ndarray:
        """Run the growth process and return the resulting output image.

        Args:
            width (int): Output image width.
            height (int): Output image height.
            steps (int, optional): Number of inference steps. Defaults to 100.

        Returns:
            np.ndarray: Image channels of the output image.
        """
        out = torch.zeros((1, self.num_channels, width, height)).to(self.device)
        out[:, 3:, :, :] = 1.0
        out = out.permute(0, 2, 3, 1)

        if save_steps:
            step_outs = []
            for _ in range(steps):
                out = self.forward(out, steps=1)
                step_outs.append(
                    np.clip(out[..., : self.num_image_channels].squeeze().detach().cpu().numpy(), 0, 1)
                )
            return step_outs
        else:
            out = self.forward(out, steps=steps)
        out_np = out[..., : self.num_image_channels].detach().cpu().numpy()[0]
        out_np = np.clip(out_np, 0, 1)
        return out_np
