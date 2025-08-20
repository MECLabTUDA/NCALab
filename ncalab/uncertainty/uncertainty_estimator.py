from typing import Tuple, List

import torch

from ..models import BasicNCAModel
from ..prediction import Prediction


class UncertaintyEstimator:
    def __init__(self, nca: BasicNCAModel):
        """
        :param nca: Trained NCA model
        :type nca: BasicNCAModel
        """
        self.nca = nca

    def _estimate(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[Prediction]]:
        """
        Internal uncertainty estimation method.

        :param image: Input image sample
        :type image: torch.Tensor

        :return: Float tensor of uncertainty heatmap (BCWH), predictions, reduced uncertainty score
        :rtype: Tuple[torch.Tensor, List[Prediction], float]
        """
        return NotImplemented

    def estimate(
        self, image: torch.Tensor, reduce: str = "mean"
    ) -> Tuple[torch.Tensor, List[Prediction], torch.Tensor]:
        """
        Estimate predictive uncertainty.

        :param image: Input image sample
        :type image: torch.Tensor
        :param reduce: Reduction strategy, defaults to "mean".
        :type reduce: str

        :return: Float tensor of uncertainty heatmap (BCWH), final prediction, reduced uncertainty score for batch (BC)
        :rtype: Tuple[torch.Tensor, List[Prediction], torch.Tensor]
        """
        assert reduce in ("mean",)
        heatmap, predictions = self._estimate(image)
        # TODO average predictions
        if reduce == "mean":
            score = torch.mean(heatmap, dim=(2, 3))
        elif reduce == "max":
            score, _ = torch.max(torch.max(image, dim=2)[0], dim=2)
        else:
            score = torch.mean(heatmap, dim=(2, 3))
        return heatmap, predictions, score

    def __call__(self, *args, **kwargs):
        self.estimate(*args, **kwargs)


class NQM(UncertaintyEstimator):
    """
    Variance over multiple predictions.
    """

    def __init__(self, nca: BasicNCAModel, N: int = 10, normalize=False):
        super().__init__(nca)
        self.N = N
        self.normalize = normalize

    def _estimate(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[Prediction]]:
        """
        Internal uncertainty estimation method.

        :param image: Input image sample
        :type image: torch.Tensor

        :return: Float tensor of uncertainty heatmap (BCWH), predictions, reduced uncertainty score
        :rtype: Tuple[torch.Tensor, List[Prediction], float]
        """
        self.nca.eval()
        heatmap = torch.zeros_like(image)
        # sample N times
        sequence = [self.nca.record(image)[-1] for _ in range(self.N)]
        output_images = torch.stack(
            [prediction.output_image for prediction in sequence]
        )

        heatmap = torch.std(
            output_images[:, :, -self.nca.num_output_channels :, :, :], dim=0
        )
        if self.normalize:
            normalization = torch.mean(
                torch.sum(
                    output_images[:, :, -self.nca.num_output_channels :, ...] > 0,
                    dim=(3, 4),
                    keepdim=True,
                    dtype=torch.float32,
                ),
                dim=0,
            )
            heatmap /= normalization
        return heatmap, sequence


class MCMC(UncertaintyEstimator):
    """
    Markov-Chain Monte Carlo
    """

    def __init__(self, nca: BasicNCAModel, N_last: int = 10, normalize=False):
        super().__init__(nca)
        self.N_last = N_last
        self.normalize = normalize

    def _estimate(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[Prediction]]:
        """
        Internal uncertainty estimation method.

        :param image: Input image sample
        :type image: torch.Tensor

        :return: Float tensor of uncertainty heatmap (BCWH), predictions, reduced uncertainty score
        :rtype: Tuple[torch.Tensor, List[Prediction], float]
        """
        self.nca.eval()
        heatmap = torch.zeros_like(image)
        sequence = self.nca.record(image)[-self.N_last :]
        output_images = torch.stack(
            [prediction.hidden_channels for prediction in sequence]
        )

        heatmap = torch.std(output_images, dim=0)
        if self.normalize:
            normalization = 10
            heatmap /= normalization
        return heatmap, sequence
