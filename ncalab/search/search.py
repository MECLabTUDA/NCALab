import itertools
from collections.abc import Iterable
import logging
from typing import Any, Dict, Optional

import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader  # for type hint

from ..training import BasicNCATrainer  # for type hint


class ParameterSet:
    """ """

    def __init__(self, **kwargs):
        """ """
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

        # compute carthesian product of all possible combinations
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

    def next(self) -> Dict[str, Any]:
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

    def info(self) -> str:
        """
        Generate information string with a summary of the search to run.
        """
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
        dataloader_val: Optional[DataLoader] = None,
    ):
        """
        Run search.

        :param dataloader_train [DataLoader]: Training DataLoader.
        :param dataloader_val [DataLoader]: Validation DataLoader. Defaults to None.
        """

        list_of_summaries = []
        i = 0
        for trainer_args in self.trainer_params:
            for model_args in self.model_params:
                # experiment index
                i += 1
                # create name for tensorboard comment
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
                # TODO: allow k-fold trainer
                trainer = BasicNCATrainer(model, **trainer_args)
                summary = trainer.train(
                    dataloader_train,
                    dataloader_val,
                    summary_writer=writer,
                )

                # save current set of (mutable) args in dict
                d = summary.to_dict()
                d.update(
                    **{
                        k: v
                        for k, v in trainer_args.items()
                        if self.trainer_params.is_mutable(k)
                    }
                )
                d.update(
                    **{
                        k: v
                        for k, v in model_args.items()
                        if self.model_params.is_mutable(k)
                    }
                )
                list_of_summaries.append(d)
                writer.close()

                # Print intermediate results
                df = pd.DataFrame(list_of_summaries)
                print(df)
        return pd.DataFrame(list_of_summaries)

    def __call__(self, *args, **kwargs):
        """
        Shorthand for running the search.
        """
        return self.search(*args, **kwargs)
