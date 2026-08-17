import torch

from ncalab import (
    ClassificationNCAModel,
    EstimatorNQM,
    get_compute_device,
)


def test_nqm():
    steps = 100
    classes = 10
    N = 12
    B, C, W, H = 8, classes + 3 + 3, 32, 32
    # TODO: test model with classifier head
    nca = ClassificationNCAModel(
        get_compute_device("cpu"),
        3,
        3,
        classes,
        use_classifier=False,
        inference_timesteps=steps,
    )
    assert nca.num_channels == C
    X = torch.normal(0, 0.22, size=(B, C, W, H))
    estimator = EstimatorNQM(nca, N=N, normalize=True)
    for reduce in ("mean", "min", "max"):
        heatmap, _, score = estimator(X, reduce=reduce)
        assert heatmap.shape == (B, classes, W, H)
        assert score.shape == (B, classes)
