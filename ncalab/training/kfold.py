import copy
import json
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Type, Optional

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .trainer import BasicNCATrainer
from .trainingsummary import TrainingSummary


class TrainValRecord:
    """
    Helper class, storing a training / validation data split to generate
    respective DataLoader objects.
    """

    def __init__(
        self,
        train: List[str],
        val: List[str],
    ):
        """
        Constructor.

        :param train (List[str]): List of training image file paths
        :param val (List[str]): List of validation image file paths
        """
        self.train = train
        self.val = val

    def dataloaders(
        self,
        DatasetType: Type,
        path: Path | PosixPath,
        transform=None,
        batch_sizes=None,
    ):
        """
        Generate a pair of training and validation DataLoader objects, based on
        a given DataSet subtype.
        """
        if batch_sizes is None:
            batch_sizes = {"train": 8, "val": 8}
        dataset_train = DatasetType(path, self.train, transform)
        dataloader_train = DataLoader(
            dataset_train, shuffle=True, drop_last=True, batch_size=batch_sizes["train"]
        )

        dataset_val = DatasetType(path, self.val, transform)
        dataloader_val = DataLoader(
            dataset_val, shuffle=True, drop_last=True, batch_size=batch_sizes["val"]
        )

        return {
            "train": dataloader_train,
            "val": dataloader_val,
        }


class SplitDefinition:
    """
    Stores a k-fold cross-validation split.
    """

    def __init__(self):
        """
        Constructor.
        """
        self.folds = []
        self.dataloader_test = None

    @staticmethod
    def read(path: PosixPath) -> "SplitDefinition":
        """
        Reads json files with split definitions, similar to those created by nnUNet.

        Format is like

        .. highlight:: python
        .. code-block:: python

            [
                {
                    "train": [ "filename0", "filename1",... ]
                    "val": [ "filename2", "filename3",... ]
                },
                {
                    ...
                }
            ]

        :param path [PosixPath]: Path to JSON file containing split definition.
        """
        with open(path, "r") as f:
            d = json.load(f)
        sd = SplitDefinition()
        sd.folds = []
        for fold in d:
            train = fold["train"]
            val = fold["val"]
            tvt = TrainValRecord(train, val)
            sd.folds.append(tvt)
        # TODO validate structure
        return sd

    def __len__(self) -> int:
        return len(self.folds)

    def __getitem__(self, idx) -> TrainValRecord:
        return self.folds[idx]


class KFoldCrossValidationTrainer:
    def __init__(self, trainer: BasicNCATrainer, split: SplitDefinition):
        """
        Constructor.

        :param trainer [BasicNCATrainer]: BasicNCATrainer, to train each individual fold.
        :param split [SplitDefinition]: Definition of the split used for k-fold cross-training.
        """
        self.trainer = trainer
        self.model_prototype = copy.deepcopy(trainer.nca)
        self.model_name = trainer.model_path.with_suffix("")
        self.split = split

    def train(
        self,
        DatasetType: Type,
        datapath: Path | PosixPath,
        transform,
        batch_sizes: None | Dict = None,
        save_every: int | None = None,
    ) -> List[TrainingSummary]:
        """
        Run training loop with a single function call.

        :param DatasetType [Type]: Type of dataset class to use.
        :param datapath [Path]: _description_
        :param transform: Data transform, e.g. initialized via Albumentations.
        :param batch_sizes: Dict of batch sizes per set, e.g. {"train": 8, "val": 16}. Defaults to None.
        :param save_every [int]: _description_. Defaults to None.
        :param plot_function: Plot function override. If None, use model's default. Defaults to None.

        :returns [List[TrainingSummary]]: List of TrainingSummary objects, one per fold.
        """
        k = len(self.split)
        summaries = []
        for i in range(k):
            experiment_name = f"{self.model_name}_fold{i:02d}.pth"
            writer = SummaryWriter(comment=experiment_name)

            dataloaders = self.split[i].dataloaders(
                DatasetType, datapath, transform, batch_sizes
            )
            self.trainer.nca = copy.deepcopy(self.model_prototype)
            self.trainer.model_path = Path(experiment_name)
            summary = self.trainer.train(
                dataloaders["train"],
                dataloaders["val"],
                save_every=save_every,
                summary_writer=writer,
            )
            summaries.append(summary)
        return summaries
