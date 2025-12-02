from typing import Dict, List, Optional, Tuple

import numpy as np
import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

from ....prediction import Prediction
from ....visualization import VisualGrowing
from ...basicNCA import BasicNCAModel


class GrowingNCAModel(BasicNCAModel):
    """
    NCA Model class for "growing" tasks, in which a structure is grown from a single seed pixel.

    This specialization of the BasicNCAModel has some interesting properties.
    For instance, it has no output channels, as the growing task directly
    manipulates the input image channels.
    """

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
            plot_function=VisualGrowing(),
            num_output_channels=0,
            fire_rate=fire_rate,
            hidden_size=hidden_size,
            use_alive_mask=use_alive_mask,
            immutable_image_channels=False,
            pad_noise=False,
            **kwargs,
        )

    def loss(self, pred: Prediction, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Implements a simple MSE loss between target and prediction.

        :param pred: Prediction
        :param label: Target

        :returns [Tensor]: MSE Loss
        """
        assert label.shape[1] == self.num_image_channels
        loss = F.mse_loss(pred.image_channels, label)
        return {"total": loss}

    def validate(
        self, image: torch.Tensor, label: torch.Tensor, steps: Optional[int] = None
    ) -> Optional[Tuple[Dict[str, float], Prediction]]:
        """
        We typically don't validate during training of Growing NCA,
        because there is only a single sample in the training set.
        """
        return None

    def make_seed(self, width: int, height: int) -> torch.Tensor:
        x = torch.zeros((1, self.num_channels, width, height)).to(self.device)
        # set seed in center
        x[:, 3:, width // 2, height // 2] = 1.0
        return x

    def grow(self, seed: torch.Tensor, steps: int = 100) -> List[np.ndarray]:
        """
        Run the growth process and return the resulting output sequence.

        :param seed [torch.Tensor]: Seed image, can be generated through make_seed.
        :param steps [int]: Number of inference steps. Defaults to 100.

        :returns [List[np.ndarray]]: Sequence of output images.
        """
        with torch.no_grad():
            self.eval()
            output = []
            sequence = self.record(seed, steps=steps)
            for prediction in sequence:
                output.append(np.clip(prediction.image_channels_np.squeeze(0), 0, 1))
            return output
