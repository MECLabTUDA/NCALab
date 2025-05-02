class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_accuracy = 0
        self.counter = 0

    def done(self):
        return self.counter >= self.patience

    def step(self, accuracy):
        self.counter += 1
        if accuracy >= self.best_accuracy + self.min_delta:
            self.best_accuracy = accuracy
            self.counter = 0
