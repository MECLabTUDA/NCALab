import torch

from .trainerhook import TrainerHook


class EarlyStopping(TrainerHook):
    """
    Early stopping helper class.
    Helps to stop the training if no change in validation metrics is observed.
    """

    def __init__(self, patience: int, min_delta: float = 1e-6):
        """
        :param patience: Steps to wait until stopping the training.
        :type patience: int
        :param min_delta: Minimum deviation until counter is reset, defaults to 1e-6.
        :type min_delta: float
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_accuracy = 0.0
        self.counter = 0

    def halt_training(self) -> bool:
        return self.counter >= self.patience

    def _step(self, accuracy: torch.Tensor | float):
        """
        Increases internal counter if accuracy doesn't improve, otherwise
        resets the counter.

        Needs to be called in training loop, once per epoch.

        :param accuracy: Validation accuracy.
        :type accuracy: float
        """
        self.counter += 1
        if accuracy >= self.best_accuracy + self.min_delta:
            self.best_accuracy = accuracy
            self.counter = 0

    def on_validate(self, validation_accuracy: torch.Tensor | float):
        self._step(validation_accuracy)
