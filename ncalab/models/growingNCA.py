from typing import Dict, List, Optional, Tuple

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

import numpy as np

from .basicNCA import AutoStepper, BasicNCAModel
from ..prediction import Prediction
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
        autostepper: Optional[AutoStepper] = None,
        **kwargs,
    ):
        """
        NCA Model class for "growing" tasks, in which a structure is grown from a single seed pixel.

        This specialization of the BasicNCAModel has some interesting properties.
        For instance, it has no output channels, as the growing task directly
        manipulates the input image channels.

        :param device [torch.device]: Pytorch device descriptor.
        :param num_image_channels [int]: Number of channels reserved for input image. Defaults to 4.
        :param num_hidden_channels [int]: Number of hidden channels (communication channels). Defaults to 16.
        :param fire_rate [float]: Stochastic weight update. Defaults to 0.5.
        :param hidden_size [int]: Default number of nodes in hidden layer. Defaults to 128.
        :param use_alive_mask [bool]: Whether to use alive masking. Defaults to False.
        """
        super(GrowingNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            plot_function=show_batch_growing,
            num_output_channels=0,
            fire_rate=fire_rate,
            hidden_size=hidden_size,
            use_alive_mask=use_alive_mask,
            immutable_image_channels=False,
            pad_noise=False,
            autostepper=autostepper,
            **kwargs,
        )

    def loss(self, image: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Implements a simple MSE loss between target and prediction.

        :param x [Tensor]: Prediction, BCWH
        :param y [Tensor]: Target, BCWH

        :returns [Tensor]: MSE Loss
        """
        assert image.shape[1] == self.num_channels
        assert label.shape[1] == self.num_image_channels
        loss = F.mse_loss(image[:, : self.num_image_channels, :, :], label)
        return {"total": loss}

    def validate(
        self, image: torch.Tensor, label: torch.Tensor, steps: int
    ) -> Optional[Tuple[Dict[str, float], Prediction]]:
        """
        We typically don't validate during training of Growing NCA,
        because there is only a single sample in the training set.
        """
        return None

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
            # TODO make use of autostepper, if available
            self.eval()
            x = torch.zeros((1, self.num_channels, width, height)).to(self.device)
            # set seed in center
            x[:, 3:, width // 2, height // 2] = 1.0

            if save_steps:
                step_outs = []
                for _ in range(steps):
                    prediction = self.forward(x, steps=1)  # type: ignore[assignment]
                    step_outs.append(
                        np.clip(
                            prediction.image_channels
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .numpy(),
                            0,
                            1,
                        )
                    )
                    x = prediction.output_image
                return step_outs
            else:
                prediction = self.forward(x, steps=steps)  # type: ignore[assignment]
            out_np = (
                prediction.image_channels.detach().cpu().numpy().squeeze(0)
            )
            out_np = np.clip(out_np, 0, 1)
            return out_np
