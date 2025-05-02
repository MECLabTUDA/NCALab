import itertools
from collections.abc import Iterable
import logging

import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader  # for type hint

from ..training import BasicNCATrainer  # for type hint


class ParameterSet:
    def __init__(self, **kwargs):
        """_summary_"""
        self.params = kwargs
        # Replace parameters by iterable parameters (list with single entry)
        # if they are not iterables.
        # Consider that strings are iterable.

        self.mutable = [
            k
            for k, v in kwargs.items()
            if isinstance(v, Iterable) and not isinstance(v, str)
        ]
        self.params = {k: v if k in self.mutable else [v] for k, v in kwargs.items()}

        C = itertools.product(*self.params.values())
        self.combinations = []
        for combination in C:
            self.combinations.append(dict(zip(self.params.keys(), combination)))
        if not self.combinations:
            self.combinations = [{}]
        self.index = 0

    def is_mutable(self, key):
        return key in self.mutable

    def info(self):
        s = ""
        if self.mutable:
            s += " × ".join(self.mutable) + "\n"
        else:
            s += "Empty set\n"
        s += f"{len(self)} combinations"
        return s

    def next(self):
        if self.index < len(self.combinations):
            ret = self.combinations[self.index]
            self.index += 1
            return ret
        raise StopIteration

    def num_combinations(self):
        return len(self.combinations)

    def __len__(self):
        return self.num_combinations()

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class ParameterSearch:
    def __init__(
        self,
        device,
        model_class,
        model_params: ParameterSet,
        trainer_params: ParameterSet,
    ):
        self.device = device
        self.model_class = model_class
        self.model_params = model_params
        self.trainer_params = trainer_params

    def info(self):
        model_info = self.model_params.info()
        model_info = "\n".join(["|    " + L for L in model_info.splitlines()])
        trainer_info = self.trainer_params.info()
        trainer_info = "\n".join(["|    " + L for L in trainer_info.splitlines()])

        s = ""
        s += "\n" + "-" * 40 + "\n"
        s += "| Hyperparameter Search Summary"
        s += "\n" + "-" * 40 + "\n"
        s += "| Model Parameters:\n"
        s += model_info
        s += "\n|\n| Trainer Parameters:\n"
        s += trainer_info
        s += "\n" + "-" * 40 + "\n"
        s += (
            f"| Total combinations: {len(self.model_params) * len(self.trainer_params)}"
        )
        s += "\n" + "-" * 40 + "\n"
        return s

    def search(
        self,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader | None = None,
    ):
        """
        Run search.

        Args:
            dataloader_train (DataLoader): _description_
            dataloader_val (DataLoader | None, optional): _description_. Defaults to None.
        """

        list_of_summaries = []
        i = 0
        for trainer_args in self.trainer_params:
            for model_args in self.model_params:
                i += 1
                experiment_name = "T_" + "".join(
                    [
                        f"_{k}={v}" if self.trainer_params.is_mutable(k) else ""
                        for k, v in trainer_args.items()
                    ]
                )
                experiment_name += "__M_" + "".join(
                    [
                        f"_{k}={v}" if self.model_params.is_mutable(k) else ""
                        for k, v in model_args.items()
                    ]
                )
                logging.info(f"Running experiment '{experiment_name}'")
                logging.info(
                    f"This is experiment {i} / {len(self.trainer_params) * len(self.model_params)}"
                )
                writer = SummaryWriter(comment=experiment_name)
                model = self.model_class(self.device, **model_args)
                # TODO: allow k-fold
                trainer = BasicNCATrainer(model, **trainer_args)
                summary = trainer.train(
                    dataloader_train,
                    dataloader_val,
                    summary_writer=writer,
                )
                d = summary.to_dict()
                d.update(**trainer_args)
                d.update(**model_args)
                list_of_summaries.append(d)
                writer.close()
        return pd.DataFrame(list_of_summaries)

    def __call__(self, *args, **kwargs):
        return self.search(*args, **kwargs)
