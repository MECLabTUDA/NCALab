import abc
from typing import Literal

import torch

from ..models import AbstractNCAModel
from ..prediction import Prediction


class UncertaintyEstimator(abc.ABC):
    def __init__(self, nca: AbstractNCAModel):
        """
        :param nca: Trained NCA model
        :type nca: AbstractNCAModel
        """
        self.nca = nca

    @abc.abstractmethod
    def _estimate(self, image: torch.Tensor) -> tuple[torch.Tensor, list[Prediction]]:
        """
        Internal uncertainty estimation method.

        :param image: Input image sample
        :type image: torch.Tensor

        :return: Float tensor of uncertainty heatmap (BCWH), predictions, reduced uncertainty score
        :rtype: Tuple[torch.Tensor, List[Prediction], float]
        """
        raise NotImplementedError

    def estimate(
        self, image: torch.Tensor, reduce: Literal["min", "max", "mean"] = "mean"
    ) -> tuple[torch.Tensor, list[Prediction], torch.Tensor]:
        """
        Estimate predictive uncertainty.

        :param image: Input image sample
        :type image: torch.Tensor
        :param reduce: Reduction strategy, defaults to "mean".
        :type reduce: str

        :return: Float tensor of uncertainty heatmap (BCWH), final prediction, reduced uncertainty score for batch (BC)
        :rtype: Tuple[torch.Tensor, List[Prediction], torch.Tensor]
        """
        assert reduce in ("mean", "min", "max")
        heatmap, predictions = self._estimate(image)
        match reduce:
            case "mean":
                score = torch.mean(heatmap, dim=(2, 3))
            case "min":
                score, _ = torch.min(torch.min(heatmap, dim=2)[0], dim=2)
            case "max":
                score, _ = torch.max(torch.max(heatmap, dim=2)[0], dim=2)
            case _:
                score = torch.mean(heatmap, dim=(2, 3))
        return heatmap, predictions, score

    def __call__(self, *args, **kwargs):
        return self.estimate(*args, **kwargs)


class EstimatorNQM(UncertaintyEstimator):
    """
    Variance over multiple predictions.
    """

    def __init__(self, nca: AbstractNCAModel, N: int = 10, normalize: bool = False):
        super().__init__(nca)
        self.N = N
        self.normalize = normalize

    def _estimate(self, image: torch.Tensor) -> tuple[torch.Tensor, list[Prediction]]:
        """
        Internal uncertainty estimation method.

        :param image: Input image sample
        :type image: torch.Tensor

        :return: Float tensor of uncertainty heatmap (BCWH), predictions, reduced uncertainty score
        :rtype: Tuple[torch.Tensor, List[Prediction], float]
        """
        self.nca.eval()
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
