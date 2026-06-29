import pytest
import torch

from ncalab import (
    GrowingNCAModel,
    fix_random_seed,
    get_compute_device,
    interpret_range_parameter,
    pad_input,
    print_mascot,
    print_NCALab_banner,
    release_random_seed,
    unwrap,
)


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


def test_print_functions():
    try:
        print_mascot("hello world")
    except Exception as e:
        pytest.fail(str(e))
    try:
        print_NCALab_banner()
    except Exception as e:
        pytest.fail(str(e))


def test_fix_random_seed():
    fix_random_seed()
    x_torch_before = torch.rand(10)
    fix_random_seed()
    x_torch_after = torch.rand(10)
    assert torch.all(x_torch_before == x_torch_after)


def test_release_random_seed():
    fix_random_seed(42)
    x_torch_before = torch.rand(1000)
    seed = release_random_seed()
    x_torch_after = torch.rand(1000)
    if seed != 42:  # very unlikely
        assert not torch.all(x_torch_before == x_torch_after)
    else:
        assert torch.all(x_torch_before == x_torch_after)


def test_unwrap():
    x = 1
    assert unwrap(x) == x
    y = object()
    assert unwrap(y) is y
    z = None
    with pytest.raises(RuntimeError):
        z = unwrap(z)


def test_interpret_range_parameter():
    x = 1
    assert interpret_range_parameter(x) == 1
    y = (1, 5)
    assert 1 <= interpret_range_parameter(y) <= 5
    with pytest.raises(TypeError):
        z = (1, 2, 3)
        interpret_range_parameter(z)
