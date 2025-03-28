from .basicNCA import BasicNCAModel
from .splitNCA import SplitNCAModel

from ..visualization import show_batch_depth
from ..utils import pad_input

import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

from pytorch_msssim import ssim  # type: ignore[import-untyped]


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth_map, rgb_image):
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


class DepthNCAModel(SplitNCAModel):
    def __init__(
        self,
        device: torch.device,
        num_image_channels: int,
        num_hidden_channels: int,
        num_classes: int,
        fire_rate: float = 0.0,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        immutable_image_channels: bool = True,
        learned_filters: int = 4,
        lambda_activity: float = 0.0,
        pad_noise: bool = False,
    ):
        """NCA model for monocular depth estimation.

        Args:
            device (device): Compute device
            num_image_channels (int): _description_
            num_hidden_channels (int): _description_
            num_classes (int): _description_
            fire_rate (float, optional): _description_. Defaults to 0.5.
            hidden_size (int, optional): _description_. Defaults to 128.
            use_alive_mask (bool, optional): _description_. Defaults to False.
            immutable_image_channels (bool, optional): _description_. Defaults to True.
            learned_filters (int, optional): _description_. Defaults to 2.
        """
        self.num_classes = num_classes
        self.lambda_activity = lambda_activity
        super(DepthNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            num_classes,
            fire_rate,
            hidden_size,
            use_alive_mask,
            immutable_image_channels,
            learned_filters,
            kernel_size=3,
            pad_noise=pad_noise,
        )
        self.plot_function = show_batch_depth
        self.vignette = None

    def prepare_input(self, x):
        # TODO: create positional encoding
        return x

    def estimate_depth(self, image, steps=80):
        with torch.no_grad():
            x = image.clone()
            x = pad_input(x, self, noise=self.pad_noise)
            x = self.prepare_input(x)
            x = x.permute(0, 2, 3, 1)
            x = self(x, steps=steps)

            class_channels = x[
                ..., self.num_image_channels + self.num_hidden_channels :
            ]

            return class_channels

    def loss(self, x, y):
        out_channels = x[..., self.num_image_channels + self.num_hidden_channels :]
        y_pred = out_channels.permute(0, 3, 1, 2).squeeze(1)

        assert y_pred.shape == y.shape
        B, W, H = y_pred.shape

        t_gt = torch.median(torch.median(y, dim=1)[0], dim=1)[0]
        t_pred = torch.median(torch.median(y_pred, dim=1)[0], dim=1)[0]

        # Scale-Shift Invariant MSE Loss
        y_SSI = y.permute(1, 2, 0) - t_gt
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
            y_pred.unsqueeze(1), y.unsqueeze(1), data_range=1.0
        )

        loss_tv_function = SmoothnessLoss().to(self.device)
        loss_tv = loss_tv_function(
            y_pred.unsqueeze(1),
            y.unsqueeze(1),
        )

        loss_depthmap_function = nn.MSELoss()
        loss_depthmap = loss_depthmap_function(
            y_pred_SSI,
            y_SSI,
        )

        loss = 0.5 * loss_tv + loss_depthmap + loss_ssim
        return {"total": loss, "tv": loss_tv, "depth": loss_depthmap, "ssim": loss_ssim}

    def validate(
        self,
        dataloader_val,
        steps: int,
        batch_iteration: int,
        summary_writer=None,
    ):
        self.eval()
        total_ssim = 0.0
        N = 0.0

        with torch.no_grad():
            for images, labels in dataloader_val:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.estimate_depth(images, steps=steps).permute(0, 3, 1, 2)

                s = ssim(outputs, labels.unsqueeze(1), data_range=1.0)

                total_ssim += s
                N += 1.0
        total_ssim /= N
        if summary_writer:
            summary_writer.add_scalar("Acc/val_acc", total_ssim, batch_iteration)
        return total_ssim
