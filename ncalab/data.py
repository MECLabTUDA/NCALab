# TODO implement pattern pool
# TODO implement damage
import numpy as np
import torch  # type: ignore[import-untyped]
from torch.utils.data import Dataset  # type: ignore[import-untyped]


class GrowingNCADataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        num_channels: int,
        batch_size: int = 8,
        use_pattern_pool: bool = False,
        damage: bool = False,
    ):
        """Dedicated dataset for "growing" tasks, like growing emoji.

        The idea is to train a model solely for the purpose to generate ("grow")
        a fixed image. Hence, this Dataset class only stores multiple copies of the
        same image.

        :param image [np.ndarray]: Input image.
        :param num_channels [int]: Total number of image channels (including hidden)
        :param batch_size [int]: Output batch size. Defaults to 8.
        """
        super(GrowingNCADataset, self).__init__()
        self.batch_size = batch_size
        self.image = image.astype(np.float32) / 255.0
        self.seed = np.zeros((num_channels, image.shape[0], image.shape[1]))
        self.seed[3:, :, :] = 1.0

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        seed = self.seed.copy()
        image = self.image.copy()
        seed = torch.from_numpy(seed).float()
        image = torch.from_numpy(image).float()
        return seed, image
