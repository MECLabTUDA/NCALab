from typing import Dict

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]

from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score  # type: ignore[import-untyped]

from .basicNCA import BasicNCAModel


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
        pixel_wise_loss: bool = False,
        filter_padding: str = "reflect",
        pad_noise: bool = False,
    ):
        """
        :param device [torch.device]: Compute device.
        :param num_image_channels [int]: _description_
        :param num_hidden_channels [int]: _description_
        :param num_classes [int]: _description_
        :param fire_rate [float]: _description_. Defaults to 0.5.
        :param hidden_size [int]: _description_. Defaults to 128.
        :param use_alive_mask [bool]: _description_. Defaults to False.
        :param immutable_image_channels [bool]: _description_. Defaults to True.
        :param learned_filters [int]: _description_. Defaults to 0.
        :param pixel_wise_loss [bool]: Whether a prediction per pixel is desired, like in self-classifying MNIST. Defaults to False.
        :param filter_padding [str]: _description_. Defaults to "reflect".
        :param pad_noise [bool]: _description_. Defaults to False.
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
        self.pixel_wise_loss = pixel_wise_loss
        self.validation_metric = "accuracy_micro"

    def classify(
        self, image: torch.Tensor, steps: int = 100, reduce: bool = False
    ) -> torch.Tensor:
        """
        Predict classification for an input image.

        :param image [torch.Tensor]: Input image.
        :param steps [int]: Inference steps. Defaults to 100.
        :param reduce [bool]: Return a single softmax probability. Defaults to False.

        :returns [torch.Tensor]: Single class index or vector of logits.
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
            y_pred = torch.mean(y_pred, dim=1)  # average W
            y_pred = torch.mean(y_pred, dim=1)  # average H
            loss_mse = (
                F.mse_loss(
                    y_pred.float(),
                    F.one_hot(target.squeeze(), num_classes=self.num_classes).float(),
                    reduction="none",
                )
            ).mean()
            loss_classification = loss_mse

        loss = loss_classification
        return {
            "total": loss,
            "classification": loss_classification,
        }

    def metrics(
        self,
        pred: torch.Tensor,
        label: torch.Tensor
    ):
        accuracy_macro_metric = MulticlassAccuracy(
            average="macro", num_classes=self.num_classes
        )
        accuracy_micro_metric = MulticlassAccuracy(
            average="micro", num_classes=self.num_classes
        )
        auroc_metric = MulticlassAUROC(num_classes=self.num_classes)
        f1_metric = MulticlassF1Score(num_classes=self.num_classes)

        y_prob = pred[..., -self.num_output_channels :]
        y_true = label.squeeze()
        accuracy_macro_metric.update(y_prob, y_true)
        accuracy_micro_metric.update(y_prob, y_true)
        auroc_metric.update(y_prob, y_true)
        f1_metric.update(y_prob, y_true)

        accuracy_macro = accuracy_macro_metric.compute().item()
        accuracy_micro = accuracy_micro_metric.compute().item()
        auroc = auroc_metric.compute().item()
        f1 = f1_metric.compute().item()
        return {
            "accuracy_macro": accuracy_macro,
            "accuracy_micro": accuracy_micro,
            "f1": f1,
            "auroc": auroc,
            "prediction": y_prob,
        }

    def get_meta_dict(self) -> dict:
        meta = super().get_meta_dict()
        meta.update(
            dict(
                num_classes=self.num_classes,
                pixel_wise_loss=self.pixel_wise_loss,
            )
        )
        return meta
