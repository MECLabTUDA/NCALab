import torch
import numpy as np


class Pool:
    """
    Sample pool that retains previous predictions. Also applies damaging patterns to
    images to increase the robustness of the trained NCA.
    """

    def __init__(self, n_seed: int = 1, damage: bool = False, p_damage: float = 0.2):
        """
        :param n_seed: How many seed images to retain, defaults to 1
        :type n_seed: int, optional
        :param damage: Whether to apply damaging patterns, defaults to False
        :type damage: bool, optional
        :param p_damage: Probability at which a damaging pattern is applied, defaults to 0.2
        :type p_damage: float, optional
        """
        assert n_seed >= 1
        self.n_seed = n_seed
        self.damage = damage
        self.batch: torch.Tensor | None = None
        self.p_damage = p_damage

    def update(self, batch: torch.Tensor):
        """
        :param batch: BCWH
        """
        self.batch = torch.clone(batch)

    def sample(self, seed: torch.Tensor) -> torch.Tensor:
        """
        :param seed: BCWH
        :return: BCWH
        """
        if self.n_seed >= len(seed):
            return seed
        if self.batch is None:
            return seed
        batch = torch.clone(seed)
        batch[self.n_seed :] = self.batch[: len(seed) - self.n_seed]
        if self.damage and np.random.random() < self.p_damage:
            # delete random rectangle
            w, h = batch.shape[2], batch.shape[3]
            rx = np.random.randint(0, w)
            ry = np.random.randint(0, h)
            rw = np.random.randint(w // 4)
            rh = np.random.randint(h // 4)
            batch[
                self.n_seed :,
                :,
                rx : np.clip(rx + rw, 0, w),
                ry : np.clip(ry + rh, 0, h),
            ] = 0
        return batch
