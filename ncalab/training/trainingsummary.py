class TrainingSummary:
    def __init__(self, best_acc, best_path, metrics):
        self.best_acc = best_acc
        self.best_path = best_path
        self.metrics = metrics

    def to_dict(self):
        return dict(
            best_acc=self.best_acc,
            best_path=self.best_path,
            **self.metrics
        )