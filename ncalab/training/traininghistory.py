import copy
import enum
from pathlib import Path, PosixPath
import json
import logging
from typing import Dict, Optional

import torch

from ..models import BasicNCAModel


class TrainingStatus(enum.Enum):
    STATUS_NONE = 0
    STATUS_RUNNING = 1
    STATUS_DONE = 2


class TrainingHistory:
    def __init__(
        self,
        path: Optional[Path | PosixPath],
        metrics: Dict[str, float],
        current_epoch: int,
        current_model: BasicNCAModel,
        # TODO current_accuracy
        best_accuracy: float = 0,
        best_epoch: int = 0,
        best_model: Optional[BasicNCAModel] = None,
        verbose: bool = True,
    ):
        # TODO keep track of accuracy development
        self.path = path
        self.metrics = metrics
        self.current_epoch = current_epoch
        self.current_model = current_model
        self.best_accuracy = best_accuracy
        self.best_epoch = best_epoch
        self.best_model = best_model
        self.verbose = verbose

    def update(
        self, epoch: int, model: BasicNCAModel, accuracy: float, overwrite: bool = False
    ):
        self.current_epoch = epoch
        self.current_model = model
        self.current_accuracy = accuracy
        if accuracy > self.best_accuracy or overwrite:
            if self.verbose and not overwrite:
                logging.info(
                    f"Accuracy improvement ({model.validation_metric}):\n"
                    + f"  {self.best_accuracy:.5f} --> {accuracy:.5f}"
                    + f"  in epoch {epoch}"
                )
            self.best_epoch = epoch
            self.best_accuracy = accuracy
            self.best_model = copy.deepcopy(model)

    def save(self):
        if self.path is None:
            return
        history_path = self.path / "history.json"
        last_model_path = self.path / "last_model.pth"
        best_model_path = self.path / "best_model.pth"
        self.path.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4, sort_keys=True)
        torch.save(self.current_model.state_dict(), last_model_path)
        if self.best_model is not None:
            torch.save(self.best_model.state_dict(), best_model_path)

    def to_dict(self) -> Dict:
        d = dict(
            path=str(self.path),
            metrics=self.metrics,
            current_epoch=self.current_epoch,
            best_acc=self.best_accuracy,
            best_epoch=self.best_epoch,
            **self.metrics,
            current_model=self.current_model.to_dict(),
        )
        if self.best_model is not None:
            d["best_model"] = self.best_model.to_dict()
        return d
