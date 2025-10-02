from typing import Dict, Optional

import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]
from pytorch_msssim import ssim  # type: ignore[import-untyped]

from ..prediction import Prediction
from ..visualization import VisualDepthEstimation
from .basicNCA import AutoStepper, BasicNCAModel

# TODO use torchmetrics ssim


class SmoothnessLoss(nn.Module):
    """ """

    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth_map: torch.Tensor, rgb_image: torch.Tensor):
        # Ensure the inputs are in the right shape
        assert depth_map.dim() == 4
        assert rgb_image.dim() == 4
        assert depth_map.shape[1] in (1, 3)
        assert rgb_image.shape[1] in (1, 3)

        sobel_x = (
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(depth_map.device)
        )

        sobel_y = (
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(depth_map.device)
        )

        # Apply Sobel filter to the depth map
        depth_grad_x = F.conv2d(
            depth_map, sobel_x.repeat(1, depth_map.size(1), 1, 1), padding=1
        )
        depth_grad_y = F.conv2d(
            depth_map, sobel_y.repeat(1, depth_map.size(1), 1, 1), padding=1
        )

        # Apply Sobel filter to the RGB image (apply to each channel)
        rgb_grad_x = F.conv2d(
            rgb_image, sobel_x.repeat(1, rgb_image.size(1), 1, 1), padding=1
        )
        rgb_grad_y = F.conv2d(
            rgb_image, sobel_y.repeat(1, rgb_image.size(1), 1, 1), padding=1
        )

        loss_x = F.l1_loss(depth_grad_x, rgb_grad_x)
        loss_y = F.l1_loss(depth_grad_y, rgb_grad_y)

        # Combine the losses
        total_loss = loss_x + loss_y
        return total_loss


class DepthNCAModel(BasicNCAModel):
    """
    NCA model for monocular depth estimation.
    """

    def __init__(
        self,
        device: torch.device,
        num_image_channels: int = 3,
        num_hidden_channels: int = 18,
        fire_rate: float = 0.8,
        hidden_size: int = 128,
        num_learned_filters: int = 2,
        autostepper: Optional[AutoStepper] = None,
        pad_noise: bool = False,
        **kwargs,
    ):
        super(DepthNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            plot_function=VisualDepthEstimation(),
            validation_metric="ssim",
            num_output_channels=1,
            fire_rate=fire_rate,
            hidden_size=hidden_size,
            use_alive_mask=False,
            immutable_image_channels=True,
            num_learned_filters=num_learned_filters,
            kernel_size=3,
            autostepper=autostepper,
            pad_noise=pad_noise,
            **kwargs,
        )
        self.vignette = None

    def loss(self, pred: Prediction, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param image: Input image, BCWH.
        :param label: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        out_channels = pred.output_channels
        y_pred = out_channels.squeeze(1)

        assert y_pred.shape == label.shape
        _, W, H = y_pred.shape

        t_gt = torch.median(torch.median(label, dim=1)[0], dim=1)[0]
        t_pred = torch.median(torch.median(y_pred, dim=1)[0], dim=1)[0]

        # Scale-Shift Invariant MSE Loss
        y_SSI = label.permute(1, 2, 0) - t_gt
        s_gt = torch.abs(y_SSI) / (W * H)
        s_gt = torch.sum(torch.sum(s_gt, dim=0), dim=0)
        # y_SSI = torch.div(y_SSI, s_gt)
        y_SSI = y_SSI.permute(2, 0, 1)

        y_pred_SSI = y_pred.permute(1, 2, 0) - t_pred
        s_pred = torch.abs(y_pred_SSI) / (W * H)
        s_pred = torch.sum(torch.sum(s_pred, dim=0), dim=0)
        # y_pred_SSI = torch.div(y_pred_SSI, s_pred) --> doesn't work, since predictions need to be > 0
        y_pred_SSI = y_pred_SSI.permute(2, 0, 1)

        ssim_function = ssim
        loss_ssim = 1 - ssim_function(
            y_pred.unsqueeze(1), label.unsqueeze(1), data_range=1.0
        )

        loss_tv_function = SmoothnessLoss().to(self.device)
        loss_tv = loss_tv_function(
            y_pred.unsqueeze(1),
            label.unsqueeze(1),
        )

        loss_depthmap_function = nn.MSELoss()
        loss_depthmap = loss_depthmap_function(
            y_pred_SSI,
            y_SSI,
        )

        loss = 0.5 * loss_tv + loss_depthmap + 0.2 * loss_ssim
        return {"total": loss, "tv": loss_tv, "depth": loss_depthmap, "ssim": loss_ssim}

    def metrics(self, pred: Prediction, label: torch.Tensor) -> Dict[str, float]:
        """
        Return dict of standard evaluation metrics.
        Needs to include special item 'prediction', containing the predicted image (all channels).

        :param pred:
        :param label: Ground truth label.

        :returns [Dict]: Dict of metrics, mapped by their names.
        """
        s = ssim(
            pred.output_channels, label.unsqueeze(1), data_range=1.0
        ).item()
        return {"ssim": s}
