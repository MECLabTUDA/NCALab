#!/usr/bin/env python3
import pytest

from torch.utils.data import DataLoader

import numpy as np

from ncalab import (
    GrowingNCADataset,
    GrowingNCAModel,
    BasicNCATrainer,
    get_compute_device,
)


def test_growing_training():
    """
    Test if a basic NCA trainer runs through for a few epochs without exception.
    """
    device = get_compute_device("cpu")

    nca = GrowingNCAModel(
        device,
        num_image_channels=4,
        num_hidden_channels=5,
        use_alive_mask=False,
    )

    image = np.zeros((16, 16, 4))
    dataset = GrowingNCADataset(image, nca.num_channels, batch_size=8)
    dataloader_train = DataLoader(dataset, batch_size=8, shuffle=False)

    try:
        trainer = BasicNCATrainer(
            nca, None, max_epochs=3
        )
    except Exception as e:
        pytest.fail(e)

    try:
        trainer.train(dataloader_train, save_every=100)
    except Exception as e:
        pytest.fail(e)
