import torch

from ncalab import get_compute_device, pad_input, GrowingNCAModel, fix_random_seed


def test_pad_input():
    device = get_compute_device()
    hidden_channels = 12
    batch_size = 8
    W = 32
    H = 32

    nca = GrowingNCAModel(device, num_hidden_channels=hidden_channels)
    X = torch.zeros((batch_size, nca.num_image_channels, W, H))

    total_channels = nca.num_image_channels + hidden_channels

    X_padded_zero = pad_input(X, nca, noise=False)
    assert X_padded_zero.shape == (batch_size, total_channels, W, H)
    assert torch.all(X_padded_zero[:, nca.num_image_channels :, :, :] == 0)
    X_padded_noise = pad_input(X, nca, noise=True)
    assert X_padded_noise.shape == (batch_size, total_channels, W, H)
    assert (
        torch.std(
            X_padded_noise[:, nca.num_image_channels :, :, :], dim=None, keepdim=False
        ).float()
        != 0
    )


def test_fix_random_seed():
    fix_random_seed()
    x_torch_before = torch.rand(10)
    fix_random_seed()
    x_torch_after = torch.rand(10)
    assert torch.all(x_torch_before == x_torch_after)
