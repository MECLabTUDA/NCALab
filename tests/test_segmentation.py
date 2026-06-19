import pytest

from torch.utils.data import DataLoader, Dataset

import numpy as np

from ncalab import (
    BinarySegmentationNCAModel,
    BasicNCATrainer,
    get_compute_device,
)


class DummyBinarySegmentationDataset(Dataset):
    def __init__(self, batch: int = 32, w: int = 16, h: int = 16):
        self.images = np.random.normal(size=(batch, 3, h, w)).astype(np.float32)
        self.masks = (np.random.normal(size=(batch, h, w)) > 0.5).astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


def test_binary_segmentation_training():
    """
    Test if a basic NCA trainer runs through for a few epochs without exception.
    """
    device = get_compute_device("cpu")
    batch_size = 8
    num_image_channels = 3
    num_hidden_channels = 4

    try:
        nca = BinarySegmentationNCAModel(
            device,
            num_image_channels=num_image_channels,
            num_hidden_channels=num_hidden_channels,
        )
    except Exception as e:
        pytest.fail(str(e))

    dataset = DummyBinarySegmentationDataset(batch_size * 4)
    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    try:
        trainer = BasicNCATrainer(nca, None, max_epochs=3)
    except Exception as e:
        pytest.fail(str(e))

    try:
        trainer.train(dataloader_train, save_every=100)
    except Exception as e:
        pytest.fail(str(e))
