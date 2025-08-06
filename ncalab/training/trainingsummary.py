import os
from typing import Dict

import pandas as pd


class TrainingSummary:
    def __init__(self, best_acc: float, best_path, best_training_loss: float, metrics):
        self.best_acc = best_acc
        self.best_path = best_path
        self.best_training_loss = best_training_loss
        self.metrics = metrics

    def load(self, path: os.PathLike | str):
        pass

    def save(self, path: os.PathLike | str):
        pass

    def to_dict(self) -> Dict:
        return dict(
            best_acc=self.best_acc,
            best_path=self.best_path,
            best_training_loss=self.best_training_loss,
            **self.metrics,
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict())
