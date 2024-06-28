# TODO implement pattern pool
# TODO implement damage
from torch.utils.data import Dataset
import numpy as np
import torch


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

        Args:
            image (np.ndarray): _description_
            num_channels (int): _description_
            batch_size (int, optional): Output batch size. Defaults to 8.
        """
        super(GrowingNCADataset, self).__init__()
        self.batch_size = batch_size
        self.seed = np.zeros((num_channels, image.shape[0], image.shape[1]))
        self.seed[3:, :, :] = 1.0
        self.image = image.astype(np.float32) / 255.0
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        seed = self.seed.copy()
        image = self.image.copy()
        seed = torch.from_numpy(seed).float()
        image = torch.from_numpy(image).float()
        return seed, image
