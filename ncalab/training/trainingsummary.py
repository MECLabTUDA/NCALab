from typing import Dict

import pandas as pd


class TrainingSummary:
    def __init__(self, best_acc, best_path, metrics):
        self.best_acc = best_acc
        self.best_path = best_path
        self.metrics = metrics

    def to_dict(self) -> Dict:
        return dict(best_acc=self.best_acc, best_path=self.best_path, **self.metrics)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict())
