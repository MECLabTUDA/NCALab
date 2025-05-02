from copy import deepcopy
from typing import Type, List

import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

from .basicNCA import BasicNCAModel


def upscale(image, scale):
    return nn.Upsample(scale_factor=scale, mode="nearest")(image)


def downscale(image, scale):
    return nn.Upsample(scale_factor=1 / scale, mode="bilinear", align_corners=True)(
        image
    )


class CascadeNCA(BasicNCAModel):
    """
    Chain multiple instances of the same NCA model, operating at different
    image scales.
    """

    def __init__(self, backbone: BasicNCAModel, scales: List[int], steps: List[int]):
        """
        Constructor.

        Args:
            backbone (Type): _description_
            scales (List[int]): List of scales to operate at, e.g. [4, 2, 1].
            steps (List[int]): List of number of NCA inference time steps.
        """
        super(CascadeNCA, self).__init__(
            backbone.device,
            backbone.num_image_channels,
            backbone.num_hidden_channels,
            backbone.num_output_channels,
            backbone.fire_rate,
            backbone.hidden_size,
            backbone.use_alive_mask,
            backbone.immutable_image_channels,
            backbone.num_learned_filters,
            backbone.dx_noise,
            use_laplace=backbone.use_laplace,
            pad_noise=backbone.pad_noise,
            autostepper=backbone.autostepper,
        )
        self.loss = backbone.loss
        self.get_meta_dict = backbone.get_meta_dict
        self.finetune = backbone.finetune
        self.prepare_input = backbone.prepare_input
        self.plot_function = backbone.plot_function
        self.validation_metric = backbone.validation_metric

        # TODO automatically copy attributes
        if hasattr(backbone, "segment"):
            self.segment = backbone.segment
            self.num_classes = backbone.num_classes

        self.backbone = backbone
        assert len(scales) == len(steps)
        assert len(scales) != 0
        assert scales[-1] == 1
        self.scales = scales
        self.steps = steps

        # initialize parameters by passing dummy data
        #dummy = torch.zeros((8, 16, 16, backbone.num_channels)).to(backbone.device)
        #backbone(dummy)
        #models = [deepcopy(backbone) for _ in scales]
        models = [backbone for _ in scales]
        self.models = nn.ModuleList(models)

    def forward(self, x, steps=1):
        x_scaled = downscale(x.permute(0, 3, 1, 2), self.scales[0]).permute(0, 2, 3, 1)
        for i, (model, scale, scale_steps) in enumerate(
            zip(self.models, self.scales, self.steps)
        ):
            x_pred = model(x_scaled, steps=scale_steps)
            if i < len(self.scales) - 1:
                x_scaled = upscale(
                    x_pred.permute(0, 3, 1, 2), scale / self.scales[i + 1]
                ).permute(0, 2, 3, 1)
                # replace input with downscaled variant of original image
                x_scaled[..., : model.num_image_channels] = downscale(
                    x[..., : model.num_image_channels].permute(0, 3, 1, 2),
                    self.scales[i + 1],
                ).permute(0, 2, 3, 1)
        return x_pred

    def validate(
        self, images, labels, steps: int
    ):
        # images: B W H C
        x_scaled = downscale(images.permute(0, 3, 1, 2), self.scales[0]).permute(
            0, 2, 3, 1
        )
        for i, (model, scale, scale_steps) in enumerate(
            zip(self.models, self.scales, self.steps)
        ):
            y_scaled = downscale(labels.unsqueeze(1), scale).squeeze(1)
            metrics, x_pred = model.validate(
                x_scaled,
                y_scaled,
                steps=scale_steps,
            )
            if i < len(self.scales) - 1:
                x_scaled = upscale(
                    x_pred.permute(0, 3, 1, 2), scale / self.scales[i + 1]
                ).permute(0, 2, 3, 1)
                # replace input with downscaled variant of original image
                x_scaled[..., : model.num_image_channels] = downscale(
                    images[..., : model.num_image_channels].permute(0, 3, 1, 2),
                    self.scales[i + 1],
                ).permute(0, 2, 3, 1)
        return metrics, x_pred
