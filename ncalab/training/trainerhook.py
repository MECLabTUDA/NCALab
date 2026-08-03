import abc

import torch

from ..prediction import Prediction


class TrainerHook(abc.ABC):
    def __init__(self):
        pass

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def process_training_prediction(self, prediction: Prediction) -> Prediction:
        return prediction

    def halt_training(self) -> bool:
        """
        Checks whether the training needs to be stopped.

        :returns: Whether to stop the training or not.
        :rtype: bool
        """
        return False

    def on_validate(self, validation_accuracy: torch.Tensor | float):
        pass
