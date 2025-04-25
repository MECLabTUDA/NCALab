from typing import Dict

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score  # type: ignore[import-untyped]

from tqdm import tqdm  # type: ignore[import-untyped]

from .basicNCA import BasicNCAModel
from ..utils import pad_input


class ClassificationNCAModel(BasicNCAModel):
    def __init__(
        self,
        device: torch.device,
        num_image_channels: int,
        num_hidden_channels: int,
        num_classes: int,
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        immutable_image_channels: bool = True,
        learned_filters: int = 0,
        lambda_activity: float = 0.0,
        pixel_wise_loss: bool = False,
        filter_padding: str = "reflect",
        pad_noise: bool = False,
    ):
        """
        Args:
            device (torch.device): Compute device.
            num_image_channels (int): _description_
            num_hidden_channels (int): _description_
            num_classes (int): _description_
            fire_rate (float, optional): _description_. Defaults to 0.5.
            hidden_size (int, optional): _description_. Defaults to 128.
            use_alive_mask (bool, optional): _description_. Defaults to False.
            immutable_image_channels (bool, optional): _description_. Defaults to True.
            learned_filters (int, optional): _description_. Defaults to 0.
            lambda_activity (float, optional): Activity loss weight, penalizing high NCA activity. Defaults to 0.0.
            pixel_wise_loss (bool, optional): Whether a prediction per pixel is desired, like in self-classifying MNIST. Defaults to False.
            filter_padding (str, optional): _description_. Defaults to "reflect".
            pad_noise (bool, optional): _description_. Defaults to False.
        """
        super(ClassificationNCAModel, self).__init__(
            device,
            num_image_channels,
            num_hidden_channels,
            num_classes,
            fire_rate,
            hidden_size,
            use_alive_mask,
            immutable_image_channels,
            learned_filters,
            filter_padding=filter_padding,
            pad_noise=pad_noise,
        )
        self.num_classes = num_classes
        self.lambda_activity = lambda_activity
        self.pixel_wise_loss = pixel_wise_loss

    def classify(
        self, image: torch.Tensor, steps: int = 100, reduce: bool = False
    ) -> torch.Tensor:
        """_summary_

        Args:
            image (torch.Tensor): Input image.
            steps (int, optional): Inference steps. Defaults to 100.
            reduce (bool, optional): Return a single softmax probability. Defaults to False.

        Returns:
            (torch.Tensor): Single class index or vector of logits.
        """
        with torch.no_grad():
            x = image.clone()
            x = self(x, steps=steps)
            hidden_channels = x[
                ...,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
            ]

            class_channels = x[
                ..., self.num_image_channels + self.num_hidden_channels :
            ]

            # mask inactive pixels
            for i in range(image.shape[0]):
                mask = torch.max(hidden_channels[i]) > 0.1
                class_channels[i] *= mask

            # if binary classification (e.g. self classifying MNIST),
            # mask away pixels with the binary image used as a mask
            if self.num_image_channels == 1:
                for i in range(image.shape[0]):
                    mask = image[i, ..., 0]
                    for c in range(self.num_classes):
                        class_channels[i, :, :, c] *= mask

            # Average over all pixels if a single categorial prediction is desired
            y_pred = F.softmax(class_channels, dim=-1)
            if not self.pixel_wise_loss:
                y_pred = torch.mean(y_pred, dim=1)
                y_pred = torch.mean(y_pred, dim=1)

            # If reduce enabled, reduce to a single scalar.
            # Otherwise, return logits of all channels as a vector.
            if reduce:
                y_pred = torch.argmax(y_pred, dim=-1)
                return y_pred
            return y_pred

    def loss(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return the classification loss. For pixel-wise ("self-classifying") problems,
        such as self-classifying MNIST, we compute the Cross-Entropy loss.
        For image-wise classification, MSE loss is returned.

        Args:
            x (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        # x: B W H C
        hidden_channels = x[..., self.num_image_channels : -self.num_output_channels]
        class_channels = x[..., self.num_image_channels + self.num_hidden_channels :]

        # Create one-hot ground truth tensor, where all pixels of the predicted class are
        # active in the respective classification channel.
        if self.pixel_wise_loss:
            y = torch.ones((x.shape[0], x.shape[1], x.shape[2])).to(self.device)
            # if binary images are classified: mask with first image channel
            if self.num_image_channels == 1:
                mask = x[..., 0] > 0
            # TODO: mask alpha channel if available
            else:
                mask = torch.Tensor([1.0])
            for i in range(x.shape[0]):
                y[i] *= target[i]
            loss_ce = (
                F.cross_entropy(
                    class_channels.permute(0, 3, 1, 2),  # B W H C --> B C W H
                    y.long(),
                    reduction="none",
                )
                * mask
            ).mean()
            loss_classification = loss_ce
        else:
            y_pred = F.softmax(class_channels, dim=-1)  # softmax along channel dim
            y_var = torch.var(y_pred.flatten(1, 2), dim=-1).mean()
            y_pred = torch.mean(y_pred, dim=1)  # average W
            y_pred = torch.mean(y_pred, dim=1)  # average H
            loss_mse = (
                F.mse_loss(
                    y_pred.float(),
                    F.one_hot(target.squeeze(), num_classes=self.num_classes).float(),
                    reduction="none",
                )
            ).mean()
            loss_classification = loss_mse + 0.0 * y_var

        # Activity loss, mildly penalizes highly active NCAs.
        # We want to enforce the NCA model to "focus" on important areas for classification,
        # so that masking away inactive pixels during inference becomes more effective.
        loss_activity = torch.sum(torch.square(hidden_channels)) / (
            x.shape[0] * x.shape[1] * x.shape[2]
        )

        loss = (
            1 - self.lambda_activity
        ) * loss_classification + self.lambda_activity * loss_activity
        return {
            "total": loss,
            "activity": loss_activity,
            "classification": loss_classification,
        }

    def metrics(
        self,
        image,
        label,
        steps: int,
    ):
        pass


    def validate(
        self,
        dataloader_val: torch.utils.data.DataLoader,
        steps: int,
    ) -> float:
        accuracy_macro_metric = MulticlassAccuracy(
            average="macro", num_classes=self.num_classes
        )
        accuracy_micro_metric = MulticlassAccuracy(
            average="micro", num_classes=self.num_classes
        )
        auroc_metric = MulticlassAUROC(num_classes=self.num_classes)
        f1_metric = MulticlassF1Score(num_classes=self.num_classes)

        for sample in tqdm(iter(dataloader_val), desc="Validation"):
            x, y = sample
            x = pad_input(x, self, noise=self.pad_noise)
            x = x.permute(0, 2, 3, 1).to(self.device)
            y = y.to(self.device)
            y_prob = self.classify(x, steps, reduce=False)
            # TODO adjust for pixel-wise problems
            y_true = y.squeeze()
            accuracy_macro_metric.update(y_prob, y_true)
            accuracy_micro_metric.update(y_prob, y_true)
            auroc_metric.update(y_prob, y_true)
            f1_metric.update(y_prob, y_true)

        accuracy_macro = accuracy_macro_metric.compute().item()
        accuracy_micro = accuracy_micro_metric.compute().item()
        auroc = auroc_metric.compute().item()
        f1 = f1_metric.compute().item()

        #if summary_writer:
        #    summary_writer.add_scalar(
        #        "Acc/val_acc_macro", accuracy_macro, batch_iteration
        #    )
        #    summary_writer.add_scalar(
        #        "Acc/val_acc_micro", accuracy_micro, batch_iteration
        #    )
        #    summary_writer.add_scalar("Acc/val_AUC", auroc, batch_iteration)
        #    summary_writer.add_scalar("Acc/val_F1", f1, batch_iteration)
        return f1

    def get_meta_dict(self) -> dict:
        meta = super().get_meta_dict()
        meta.update(
            dict(
                num_classes=self.num_classes,
                lambda_activity=self.lambda_activity,
                pixel_wise_loss=self.pixel_wise_loss,
            )
        )
        return meta
