
import torch

from ..prediction import Prediction
from .trainerhook import TrainerHook


class Pool(TrainerHook):
    """
    Sample pool that retains previous predictions.
    """

    def __init__(self, n_seed: int = 1):
        """
        :param n_seed: How many seed images to retain, defaults to 1
        :type n_seed: int, optional
        :param p_damage: Probability at which a damaging pattern is applied, defaults to 0.2
        :type p_damage: float, optional
        """
        assert n_seed >= 1
        self.n_seed = n_seed
        self.batch: torch.Tensor | None = None

    def _update(self, batch: torch.Tensor, losses: torch.Tensor | None = None):
        """
        :param batch: BCWH
        :type batch: Tensor
        """
        with torch.no_grad():
            self.batch = torch.clone(batch)
            if losses is None:
                return
            _, indices = torch.sort(losses, descending=True)
            self.batch = self.batch[indices]

    def _sample(self, seed: torch.Tensor) -> torch.Tensor:
        """
        :param seed: BCWH image
        :type: Tensor
        :return: BCWH image
        """
        if self.n_seed >= len(seed):
            return seed
        if self.batch is None:
            return seed
        batch = torch.clone(seed)
        batch[self.n_seed :] = self.batch[: len(seed) - self.n_seed]
        return batch

    def prepare_input(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._sample(x), y

    def process_training_prediction(
        self, prediction: Prediction, losses: torch.Tensor
    ) -> Prediction:
        self._update(prediction.output_image, losses)
        return prediction
