class EarlyStopping:
    """
    Early stopping helper class.
    Helps to stop the training if no change in validation metrics is observed.
    """

    def __init__(self, patience: int, min_delta: float = 1e-6):
        """
        Constructor.

        :param patience [int]: Steps to wait until stopping the training.
        :param min_delta [float]: Minimum deviation until counter is reset. Defaults to 1e-6.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_accuracy = 0.0
        self.counter = 0

    def done(self):
        """
        Checks whether the training can be stopped.

        Needs to be queried in training loop, once per epoch.

        :returns [bool]: Whether to stop the training or not.
        """
        return self.counter >= self.patience

    def step(self, accuracy: float):
        """
        Increases internal counter if accuracy doesn't improve, otherwise
        resets the counter.

        Needs to be called in training loop, once per epoch.

        :param accuracy [float]: Validation accuracy.
        """
        self.counter += 1
        if accuracy >= self.best_accuracy + self.min_delta:
            self.best_accuracy = accuracy
            self.counter = 0
