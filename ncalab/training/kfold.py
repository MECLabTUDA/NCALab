import copy
import json
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Type, Optional

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .trainer import BasicNCATrainer


class TrainValTestRecord:
    def __init__(
        self,
        train: List[str],
        val: List[str],
        test: Optional[List[str]] = None,
    ):
        self.train = train
        self.val = val
        self.test = test

    def dataloaders(
        self,
        DatasetType: Type,
        path: Path | PosixPath,
        transform=None,
        batch_sizes=None,
    ):
        if batch_sizes is None:
            batch_sizes = {"train": 8, "val": 8, "test": 32}
        dataset_train = DatasetType(path, self.train, transform)
        dataloader_train = DataLoader(
            dataset_train, shuffle=True, drop_last=True, batch_size=batch_sizes["train"]
        )

        dataset_val = DatasetType(path, self.val, transform)
        dataloader_val = DataLoader(
            dataset_val, shuffle=True, drop_last=True, batch_size=batch_sizes["val"]
        )

        dataset_test = None
        if self.test:
            dataset_test = DatasetType(path, self.test, transform)
            dataloader_test = DataLoader(dataset_test, batch_size=batch_sizes["test"])

        return {
            "train": dataloader_train,
            "val": dataloader_val,
            "test": dataloader_test,
        }


class SplitDefinition:
    def __init__(self):
        self.folds = []

    @staticmethod
    def read(path: PosixPath):
        """
        Reads json files with split definitions, similar to those created by nnUNet.

        Format is like
        [
            {
                "train": [ "filename0", "filename1",... ]
                "val": [ "filename2", "filename3",... ]
            },
            {
                ...
            }
        ]

        Args:
            path (PosixPath): Path to JSON file containing split definition.
        """
        with open(path, "r") as f:
            d = json.load(f)
        sd = SplitDefinition()
        sd.folds = []
        for fold in d:
            train = d["train"]
            val = d["val"]
            test = d.get("test")
            tvt = TrainValTestRecord(train, val, test)
            sd.folds.append(tvt)
        # TODO validate structure
        return sd

    def __len__(self):
        return len(self.folds)

    def __getitem__(self, idx) -> TrainValTestRecord:
        return self.folds[idx]


class KFoldCrossValidationTrainer:
    def __init__(self, trainer: BasicNCATrainer, split: SplitDefinition, k=5):
        self.trainer = trainer
        self.k = k
        self.model_prototype = copy.deepcopy(trainer.nca)
        self.model_name = trainer.model_path
        self.split = split

    def train(
        self,
        DatasetType: Type,
        datapath: Path | PosixPath,
        transform,
        batch_sizes: None | Dict = None,
        save_every: int | None = None,
    ):
        for i in range(self.k):
            experiment_name = f"{self.model_name}_fold{self.k:02d}"
            writer = SummaryWriter(comment=experiment_name)

            dataloaders = self.split[i].dataloaders(
                DatasetType, datapath, transform, batch_sizes
            )
            self.trainer.nca = copy.deepcopy(self.model_prototype)
            self.trainer.model_path = experiment_name
            summary = self.trainer.train(
                dataloaders["train"],
                dataloaders["val"],
                dataloaders["test"],
                save_every=save_every,
                summary_writer=writer,
            )
