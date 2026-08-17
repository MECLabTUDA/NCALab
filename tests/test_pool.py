import torch

from ncalab import Pool


def test_pool():
    pool = Pool(n_seed=1)
    X = torch.zeros((8, 12, 32, 32))
    S = torch.ones_like(X)
    batch = pool._sample(S)
    assert batch.shape == S.shape
    assert torch.all(batch.isclose(S))
    pool._update(X)
    batch = pool._sample(S)
    assert torch.all(batch[0].isclose(S[0]))
    assert torch.all(batch[1:].isclose(X[1:]))
