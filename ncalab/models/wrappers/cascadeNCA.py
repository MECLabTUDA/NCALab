import copy
from typing import List, Optional, Tuple

import numpy as np
import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]

from ncalab.models.basicNCA import AbstractNCAModel
from ncalab.prediction import Prediction
from ncalab.utils import unwrap


def upscale(image: torch.Tensor, scale: float, mode: str = "nearest") -> torch.Tensor:
    """
    Upsamples an image.

    :param image: Image tensor in BCWH order.
    :type image: torch.Tensor
    :param scale: Scale factor. Must be >= 1.0
    :type scale: float
    :param mode: Interpolation mode, defaults to "nearest".
    :type mode: str

    :return: Image tensor in BCWH order.
    :rtype: torch.Tensor
    """
    assert scale >= 1.0
    x = nn.Upsample(scale_factor=scale, mode=mode)(image)
    return x


def downscale(
    image: torch.Tensor, scale: float, mode: str = "bilinear"
) -> torch.Tensor:
    """
    Downsamples an image.

    :param image: Image tensor in BCWH order.
    :type image: torch.Tensor
    :param scale: Scale factor, must be >= 1.0
    :type scale: float
    :param mode: Interpolation mode, defaults to "bilinear".

    :return: Image tensor in BCWH order.
    :rtype: torch.Tensor
    """
    assert scale >= 1.0
    x = nn.Upsample(scale_factor=1.0 / scale, mode=mode, align_corners=True)(image)
    return x


class CascadeNCA(AbstractNCAModel):
    """
    Chain multiple instances of the same NCA model, operating at different
    image scales.

    The idea is to use this model as a wrapper and drop-in replacement for an existing model.
    For instance, if we created a model `nca = SegmentationNCA(...)` and all code to interface with it,
    we could instead write `cascade = CascadeNCA(SegmentationNCA(...), scales, steps)` without the need for adjusting
    any of the interfacing code.

    This is still highly experimental. In the future, we'll work on a cleaner interface for this.
    """

    def __init__(
        self,
        wrapped: AbstractNCAModel,
        scales: List[int],
        steps: List[int],
        single_model: bool = True,
    ):
        """
        :param wrapped: Backbone model based on AbstractNCAModel.
        :type wrapped: ncalab.AbstractNCAModel
        :param scales: List of scales to operate at, e.g. [4, 2, 1].
        :type scales: List[int]
        :param steps: List of number of NCA inference time steps.
        :type steps: List[int]
        :param single_model: Only train a single instance of the NCA model
        :type single_model: bool
        """
        super(CascadeNCA, self).__init__(
            device=wrapped.device,
            num_image_channels=wrapped.num_image_channels,
            num_hidden_channels=wrapped.num_hidden_channels,
            num_output_channels=wrapped.num_output_channels,
            fire_rate=wrapped.fire_rate,
            hidden_size=wrapped.hidden_size,
            use_alive_mask=wrapped.use_alive_mask,
            immutable_image_channels=wrapped.immutable_image_channels,
            num_learned_filters=wrapped.num_learned_filters,
            plot_function=wrapped.plot_function,
            validation_metric=wrapped.validation_metric,
            use_laplace=wrapped.use_laplace,
            pad_noise=wrapped.pad_noise,
            use_temporal_encoding=wrapped.use_temporal_encoding,
            training_timesteps=wrapped.training_timesteps,
            inference_timesteps=wrapped.inference_timesteps,
        )
        self.loss = wrapped.loss  # type: ignore[method-assign]
        self.finetune = wrapped.finetune  # type: ignore[method-assign]
        self.prepare_input = wrapped.prepare_input  # type: ignore[method-assign]
        self.head = wrapped.head

        # TODO automatically copy attributes
        if hasattr(wrapped, "num_classes"):
            self.num_classes = wrapped.num_classes
        if hasattr(wrapped, "avg_pool_size"):
            self.avg_pool_size = wrapped.avg_pool_size
        if hasattr(wrapped, "class_names"):
            self.class_names = wrapped.class_names

        self.metrics = wrapped.metrics

        self.wrapped = wrapped
        assert len(scales) == len(steps)
        assert len(scales) != 0
        for i, scale in enumerate(scales):
            assert scale > 0, f"Scale {i} must be > 0, is {scale}."
        assert scales[-1] == 1
        self.scales = scales
        self.steps = steps

        self.single_model = single_model
        self.models: List[AbstractNCAModel]
        if single_model:
            self.models = [wrapped for _ in scales]
        else:
            self.models = [copy.deepcopy(wrapped) for _ in scales]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Prediction:
        """
        :param x: Input image tensor, BCWH.
        :type x: torch.Tensor
        :param steps: Unused, as steps are defined in constructor.
        :type steps: torch.Tensor

        :return: Prediction object
        :rtype: Prediction
        """
        assert len(self.scales) > 0
        assert len(self.models) > 0
        assert len(self.steps) > 0
        x_scaled = downscale(x, self.scales[0])
        total_steps = 0
        for i, (model, scale, scale_steps) in enumerate(
            zip(self.models, self.scales, self.steps)
        ):
            steps = scale_steps + np.random.randint(
                -int(scale_steps * 0.2), int(scale_steps * 0.2) + 1
            )
            if steps <= 0:
                steps = 1

            for s in range(steps):
                x_scaled = model._forward_step(x_scaled, step=total_steps)
                total_steps += 1

            if i == len(self.scales) - 1:
                continue

            x_scaled = upscale(x_scaled, scale / self.scales[i + 1])
            # replace input with downscaled variant of original image
            x_scaled[:, : model.num_image_channels, :, :] = downscale(
                x[:, : model.num_image_channels, :, :],
                self.scales[i + 1],
            )
        head_prediction = None
        logits = x_scaled[:, -self.num_output_channels :, :, :]
        if self.head is not None:
            hidden = x_scaled[
                :,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
                :,
                :,
            ]
            head_prediction = self.head(hidden)
            logits = head_prediction
        return self.wrapped.post_prediction(
            Prediction(self, total_steps, x_scaled, logits, head_prediction)
        )

    def record(
        self, image: torch.Tensor, steps: Optional[int | Tuple[int, int]] = None
    ) -> List[Prediction]:
        """
        Records predictions for all time steps and returns the resulting
        sequence of predictions.

        Takes care of scaling the image in between steps.

        :param image: Input image, BCWH.
        :type image: torch.Tensor

        :returns: List of Prediction objects.
        :rtype: List[Prediction]
        """
        assert image.shape[1] <= self.num_channels
        self.eval()
        sequence = []
        x_scaled = downscale(image, self.scales[0])
        prediction = None
        for i, (model, scale, scale_steps) in enumerate(
            zip(self.models, self.scales, self.steps)
        ):
            x_in = x_scaled
            subseq = model.record(x_in, steps=scale_steps)
            prediction = subseq[-1]
            sequence.extend(subseq)
            x_in = prediction.output_image
            if i < len(self.scales) - 1:
                x_scaled = upscale(
                    unwrap(prediction).output_image, scale / self.scales[i + 1]
                )
                # replace input with downscaled variant of original image
                x_scaled[:, : model.num_image_channels, :, :] = downscale(
                    image[:, : model.num_image_channels, :, :],
                    self.scales[i + 1],
                )
        return sequence
