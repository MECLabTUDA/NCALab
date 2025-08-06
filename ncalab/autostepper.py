import logging

import torch


class AutoStepper:
    """
    Helps selecting number of timesteps based on NCA activity.
    """

    def __init__(
        self,
        min_steps: int = 10,
        max_steps: int = 100,
        plateau: int = 5,
        verbose: bool = False,
        threshold: float = 1e-2,
    ):
        """
        Constructor.

        :param min_steps [int]: Minimum number of timesteps to always execute. Defaults to 10.
        :param max_steps [int]: Terminate after maximum number of steps. Defaults to 100.
        :param plateau [int]: _description_. Defaults to 5.
        :param verbose [bool]: Whether to log interruption to stdout. Defaults to False.
        :param threshold [float]: Score threshold. Defaults to 1e-2.
        """
        assert min_steps >= 1
        assert plateau >= 1
        assert max_steps > min_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.plateau = plateau
        self.verbose = verbose
        self.threshold = threshold
        self.cooldown = 0
        # invariant: auto_min_steps > 0, so both of these will be defined when used
        self.hidden_i: torch.Tensor | None = None
        self.hidden_i_1: torch.Tensor | None = None

    def score(self) -> torch.Tensor:
        assert self.hidden_i is not None
        assert self.hidden_i_1 is not None
        # normalized absolute difference between two hidden states
        return (self.hidden_i - self.hidden_i_1).abs().sum() / torch.numel(
            self.hidden_i
        )

    def check(self, step: int) -> bool:
        """
        Checks whether to interrupt inference after the current step.

        :param score [int]: Current NCA inference step.
        :return [bool]: Whether to interrupt inference after the current step.
        """
        with torch.no_grad():
            if step < self.min_steps:
                return False
            if step >= self.max_steps:
                return True
            if self.hidden_i is None or self.hidden_i_1 is None:
                return False
            if self.score() >= self.threshold:
                self.cooldown = 0
            else:
                self.cooldown += 1
            if self.cooldown >= self.plateau:
                if self.verbose:
                    logging.info(f"Breaking after {step} steps.")
                return True
            return False
