from typing import Optional

import torch


class Pool:
    """
    Sample pool that retains previous predictions.
    """

    def __init__(self, n_seed: int = 1):
        """
        :param n_seed: How many seed images to retain, defaults to 1
        :type n_seed: int, optional
        :param p_damage: Probability at which a damaging pattern is applied, defaults to 0.2
        :type p_damage: float, optional
        """
        assert n_seed >= 1
        self.n_seed = n_seed
        self.batch: torch.Tensor | None = None
        self.losses: torch.Tensor | None = None

    def update(self, batch: torch.Tensor, losses: Optional[torch.Tensor] = None):
        """
        :param batch: BCWH
        :type batch: Tensor
        """
        self.batch = torch.clone(batch)
        self.losses = losses

    def sample(self, seed: torch.Tensor) -> torch.Tensor:
        """
        :param seed: BCWH image
        :type: Tensor
        :return: BCWH image
        """
        # TODO rank by loss
        if self.n_seed >= len(seed):
            return seed
        if self.batch is None:
            return seed
        batch = torch.clone(seed)
        batch[self.n_seed :] = self.batch[: len(seed) - self.n_seed]
        return batch
