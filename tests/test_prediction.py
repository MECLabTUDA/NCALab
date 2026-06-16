import pytest
import torch

from ncalab.prediction import Prediction


class DummyModel:
    def __init__(self):
        self.num_image_channels = 3
        self.num_hidden_channels = 12
        self.num_output_channels = 4

    @property
    def num_channels(self):
        return (
            self.num_image_channels
            + self.num_hidden_channels
            + self.num_output_channels
        )


@pytest.fixture
def prediction():
    model = DummyModel()

    output_image = torch.arange(
        2 * model.num_channels * 4 * 4,
        dtype=torch.float32,
    ).reshape(2, model.num_channels, 4, 4)

    logits = torch.randn(2, 1, 4, 4)
    head_prediction = torch.randn(2, 5)
    mask = torch.randint(0, 2, (2, 4, 4), dtype=torch.bool)

    return Prediction(
        model=model,
        steps=100,
        output_image=output_image,
        logits=logits,
        head_prediction=head_prediction,
        mask=mask,
    )


def test_unwrap_batch(prediction: Prediction):
    items = prediction.unwrap_batch()

    assert len(items) == 2

    for item in items:
        assert item.output_image.shape[0] == 1
        assert item.logits.shape[0] == 1

    assert torch.equal(
        items[0].output_image,
        prediction.output_image[0:1],
    )

    assert torch.equal(
        items[1].output_image,
        prediction.output_image[1:2],
    )
