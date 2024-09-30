import torch

from nca import get_compute_device, pad_input, GrowingNCAModel


def test_pad_input():
    device = get_compute_device()
    image_channels = 4
    hidden_channels = 12
    batch_size = 8
    W = 32
    H = 32
    total_channels = image_channels + hidden_channels
    nca = GrowingNCAModel(device, 4, 12)
    X = torch.zeros((batch_size, image_channels, W, H))

    X_padded_zero = pad_input(X, nca, noise=False)
    assert X_padded_zero.shape == (batch_size, total_channels, W, H)
    assert torch.all(X_padded_zero[:, image_channels:, :, :] == 0)
    X_padded_noise = pad_input(X, nca, noise=True)
    assert X_padded_noise.shape == (batch_size, total_channels, W, H)
    assert (
        torch.std(X_padded_noise[:, image_channels:, :, :], dim=None, keepdim=False).float()
        != 0
    )
