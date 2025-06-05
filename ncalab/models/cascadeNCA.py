from typing import List

import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]

from .basicNCA import BasicNCAModel


def upscale(image: torch.Tensor, scale: float):
    return nn.Upsample(scale_factor=scale, mode="nearest")(image)


def downscale(image: torch.Tensor, scale: float):
    return nn.Upsample(scale_factor=1 / scale, mode="bilinear", align_corners=True)(
        image
    )


class CascadeNCA(BasicNCAModel):
    """
    Chain multiple instances of the same NCA backbone model, operating at different
    image scales.
    """

    def __init__(self, backbone: BasicNCAModel, scales: List[int], steps: List[int]):
        """
        Constructor.

        :param backbone [BasicNCAModel]: Backbone model based on BasicNCAModel.
        :param scales [List[int]]: List of scales to operate at, e.g. [4, 2, 1].
        :param steps [List[int]]: List of number of NCA inference time steps.
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
        self.loss = backbone.loss  # type: ignore[method-assign]
        self.get_meta_dict = backbone.get_meta_dict  # type: ignore[method-assign]
        self.finetune = backbone.finetune  # type: ignore[method-assign]
        self.prepare_input = backbone.prepare_input  # type: ignore[method-assign]
        self.plot_function = backbone.plot_function  # type: ignore[method-assign]
        self.validation_metric = backbone.validation_metric  # type: ignore[method-assign]

        # TODO automatically copy attributes
        if hasattr(backbone, "num_classes"):
            self.num_classes = backbone.num_classes

        self.backbone = backbone
        assert len(scales) == len(steps)
        assert len(scales) != 0
        assert scales[-1] == 1
        self.scales = scales
        self.steps = steps

        models = [backbone for _ in scales]
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        :param x [torch.Tensor]: Input image tensor, BCWH.
        :param steps [int]: Unused, as steps are defined in constructor.
        """
        x_scaled = downscale(x, self.scales[0])
        for i, (model, scale, scale_steps) in enumerate(
            zip(self.models, self.scales, self.steps)
        ):
            x_pred = model(x_scaled, steps=scale_steps)  # BWHC
            if i < len(self.scales) - 1:
                x_scaled = upscale(
                    x_pred.permute(0, 3, 1, 2), scale / self.scales[i + 1]
                )
                # replace input with downscaled variant of original image
                x_scaled[:, : model.num_image_channels, :, :] = downscale(
                    x[:, : model.num_image_channels, :, :],
                    self.scales[i + 1],
                )
        return x_pred

    def record_steps(self, x: torch.Tensor):
        step_outputs = []
        x_scaled = downscale(x, self.scales[0])
        for i, (model, scale, scale_steps) in enumerate(
            zip(self.models, self.scales, self.steps)
        ):
            x_in = x_scaled
            for _ in range(scale_steps):
                x_pred = model(x_in, steps=1)  # BWHC
                step_outputs.append(
                    upscale(x_pred.permute(0, 3, 1, 2), scale).permute(0, 2, 3, 1)
                )
                x_in = x_pred.permute(0, 3, 1, 2)
            if i < len(self.scales) - 1:
                x_scaled = upscale(
                    x_pred.permute(0, 3, 1, 2), scale / self.scales[i + 1]
                )
                # replace input with downscaled variant of original image
                x_scaled[:, : model.num_image_channels, :, :] = downscale(
                    x[:, : model.num_image_channels, :, :],
                    self.scales[i + 1],
                )

        return step_outputs

    def validate(self, image: torch.Tensor, label: torch.Tensor, steps: int = 1):
        """
        Validation method.

        Takes care of scaling the input image and label on each scale,
        and calls the respective validation method of the backbone.

        :param image [torch.Tensor]: Input image tensor, BCWH.
        :param label [torch.Tensor]: Ground truth.
        :param steps [int]: Unused, as steps are defined in constructor.

        :returns:
        """
        x_scaled = downscale(image, self.scales[0])
        for i, (model, scale, scale_steps) in enumerate(
            zip(self.models, self.scales, self.steps)
        ):
            y_scaled = downscale(label.unsqueeze(1), scale).squeeze(1)
            metrics, x_pred = model.validate(
                x_scaled,
                y_scaled,
                steps=scale_steps,
            )
            if i < len(self.scales) - 1:
                x_scaled = upscale(
                    x_pred.permute(0, 3, 1, 2), scale / self.scales[i + 1]
                )
                # replace input channel with downscaled variant of original image
                x_scaled[:, : model.num_image_channels, :, :] = downscale(
                    image[:, : model.num_image_channels, :, :],
                    self.scales[i + 1],
                )
        return metrics, x_pred
