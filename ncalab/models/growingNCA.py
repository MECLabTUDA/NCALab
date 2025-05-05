from typing import List

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

import numpy as np

from .basicNCA import BasicNCAModel
from ..visualization import show_batch_growing


class GrowingNCAModel(BasicNCAModel):
    def __init__(
        self,
        device: torch.device,
        num_image_channels: int = 4,
        num_hidden_channels: int = 16,
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        **kwargs,
    ):
        """
        NCA Model class for "growing" tasks.

        This specialization of the BasicNCAModel has some interesting properties.
        For instance, it has no output channels, as the growing task directly
        manipulates the input image channels.

        :param device [torch.device]: Pytorch device descriptor.
        :param num_image_channels [int]: Number of channels reserved for input image. Defaults to 4.
        :param num_hidden_channels [int]: Number of hidden channels (communication channels). Defaults to 16.
        :param fire_rate [float]: _description_. Defaults to 0.5.
        :param hidden_size [int]: _description_. Defaults to 128.
        :param use_alive_mask [bool]: _description_. Defaults to False.
        """
        super(GrowingNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            num_output_channels=0,
            fire_rate=fire_rate,
            hidden_size=hidden_size,
            use_alive_mask=use_alive_mask,
            immutable_image_channels=False,
            pad_noise=False,
            **kwargs,
        )
        self.plot_function = show_batch_growing

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        """
        Implements a simple MSE loss between target and prediction.

        :param x [Tensor]: Prediction
        :param y [Tensor]: Target

        :returns [Tensor]: MSE Loss
        """
        loss = F.mse_loss(x[..., : self.num_image_channels], y)
        return {"total": loss}

    def validate(self, *args, **kwargs):
        """
        We typically don't validate during training of Growing NCA,
        because there is only a single sample in the training set.
        """
        pass

    def grow(
        self, width: int, height: int, steps: int = 100, save_steps=False
    ) -> np.ndarray | List[np.ndarray]:
        """
        Run the growth process and return the resulting output image.

        :param width [int]: Output image width.
        :param height [int]: Output image height.
        :param steps [int]: Number of inference steps. Defaults to 100.

        :returns [np.ndarray]: Image channels of the output image.
        """
        with torch.no_grad():
            x = torch.zeros((1, width, height, self.num_channels)).to(self.device)
            x[:, width // 2, height // 2, 3:] = 1.0

            if save_steps:
                step_outs = []
                for _ in range(steps):
                    x = self.forward(x, steps=1)  # type: ignore[assignment]
                    step_outs.append(
                        np.clip(
                            x[..., : self.num_image_channels]
                            .squeeze()
                            .detach()
                            .cpu()
                            .numpy(),
                            0,
                            1,
                        )
                    )
                return step_outs
            else:
                x = self.forward(x, steps=steps)  # type: ignore[assignment]
            out_np = x[..., : self.num_image_channels].detach().cpu().numpy().squeeze(0)
            out_np = np.clip(out_np, 0, 1)
            return out_np
