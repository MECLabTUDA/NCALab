# TODO implement pattern pool
# TODO implement damage
from torch.utils.data import Dataset
import numpy as np


class GrowingNCADataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        num_channels: int,
        batch_size: int = 8,
        use_pattern_pool: bool = False,
        damage: bool = False,
    ):
        """_summary_

        Args:
            image (np.ndarray): _description_
            num_channels (int): _description_
            batch_size (int, optional): _description_. Defaults to 8.
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
        return self.seed, self.image
