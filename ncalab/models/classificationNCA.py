from typing import Dict, Optional, List

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]
from torch import nn

import torchmetrics
import torchmetrics.classification

from ..autostepper import AutoStepper
from .basicNCA import BasicNCAModel
from ..utils import pad_input
from ..visualization import (
    VisualBinaryImageClassification,
    VisualRGBImageClassification,
)
from ..prediction import Prediction


class ClassificationNCAModel(BasicNCAModel):
    def __init__(
        self,
        device: torch.device,
        num_image_channels: int,
        num_hidden_channels: int,
        num_classes: int,
        fire_rate: float = 0.8,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        pixel_wise_loss: bool = False,
        num_learned_filters: int = 2,
        filter_padding: str = "reflect",
        use_laplace: bool = False,
        kernel_size: int = 3,
        pad_noise: bool = False,
        autostepper: Optional[AutoStepper] = None,
        use_temporal_encoding: bool = False,
        use_classifier: bool = True,
        class_names: Optional[List[str]] = None,
        avg_pool_size: int = 3,
        **kwargs,
    ):
        """
        :param device: Pytorch device descriptor.
        :param num_image_channels: _description_
        :param num_hidden_channels: _description_
        :param num_classes: _description_
        :param fire_rate: Fire rate for stochastic weight update. Defaults to 0.8.
        :param hidden_size: Number of neurons in hidden layer. Defaults to 128.
        :param use_alive_mask: Whether to use alive masking (channel 3) during training. Defaults to False.
        :param pixel_wise_loss: Whether a prediction per pixel is desired, like in self-classifying MNIST. Defaults to False.
        :param num_learned_filters: Number of learned filters. If zero, use two sobel filters instead. Defaults to 2.
        :param filter_padding [str]: Padding type to use. Might affect reliance on spatial cues. Defaults to "circular".
        :param pad_noise [bool]: Whether to pad input image tensor with noise in hidden / output channels
        """
        super(ClassificationNCAModel, self).__init__(
            device=device,
            num_image_channels=num_image_channels,
            num_hidden_channels=num_hidden_channels,
            num_output_channels=num_classes,
            fire_rate=fire_rate,
            hidden_size=hidden_size,
            use_alive_mask=use_alive_mask,
            immutable_image_channels=True,
            plot_function=None,
            num_learned_filters=num_learned_filters,
            validation_metric="accuracy_macro",
            filter_padding=filter_padding,
            use_laplace=use_laplace,
            kernel_size=kernel_size,
            pad_noise=pad_noise,
            autostepper=autostepper,
            use_temporal_encoding=use_temporal_encoding,
        )
        self._num_classes = num_classes
        self.pixel_wise_loss = pixel_wise_loss
        self.use_classifier = use_classifier
        assert avg_pool_size >= 1
        self.avg_pool_size = avg_pool_size
        if use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(
                    self.num_hidden_channels * self.avg_pool_size**2,
                    128,
                ),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, num_classes, bias=False),
            ).to(self.device)
        if class_names is None:
            self.class_names = [str(i) for i in range(num_classes)]
        else:
            assert (
                len(class_names) == num_classes
            ), "Length of class names list must match number of classes"
            self.class_names = class_names
        if num_image_channels <= 1:
            self.plot_function = VisualBinaryImageClassification()
        else:
            self.plot_function = VisualRGBImageClassification()

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @num_classes.setter
    def num_classes(self, x: int):
        self._num_classes = x
        self.num_output_channels = x

    def classify(
        self, image: torch.Tensor, steps: int = 100, reduce: bool = False
    ) -> torch.Tensor:
        """
        Predict classification for an input image.

        :param image: Input image.
        :param steps: Inference steps. Defaults to 100.
        :param reduce: Return a single softmax probability. Defaults to False.

        :returns: Single class index or vector of logits.
        """
        with torch.no_grad():
            x = image.clone()
            x = pad_input(x, self, noise=self.pad_noise)
            prediction = self(x, steps=steps)
            class_channels = prediction.output_channels

            # if binary classification (e.g. self classifying MNIST),
            # mask away pixels with the binary image used as a mask
            if self.num_image_channels == 1:
                for i in range(image.shape[0]):
                    mask = image[i, 0, :, :]
                    for c in range(self.num_classes):
                        class_channels[i, c, :, :] *= mask

            if self.use_classifier:
                return class_channels[:, :, 0, 0]

            # Average over all pixels if a single categorial prediction is desired
            y_pred = F.softmax(class_channels, dim=1)
            y_pred = torch.mean(y_pred, dim=(2, 3))

            # If reduce enabled, reduce to a single scalar.
            # Otherwise, return logits of all channels as a vector.
            if reduce:
                y_pred = torch.argmax(y_pred, dim=0)
                return y_pred
            return y_pred

    def forward(
        self,
        x: torch.Tensor,
        steps: int = 1,
    ):
        assert x.shape[1] == self.num_channels
        x[:, -self.num_output_channels :, :, :] *= 0
        for step in range(steps):
            x = self._forward_step(x, step)
        x[:, -self.num_output_channels :, :, :] *= 0

        if self.use_classifier:
            hidden = x[:, self.num_image_channels : -self.num_output_channels, :, :]
            z = F.adaptive_avg_pool2d(
                hidden, (self.avg_pool_size, self.avg_pool_size)
            ).flatten(1, 3)
            classification = self.classifier(z).unsqueeze(-1).unsqueeze(-1)
            w, h = x.shape[2], x.shape[3]
            classification = classification.expand(-1, -1, w, h)
            x = torch.cat(
                (
                    x[:, : self.num_image_channels + self.num_hidden_channels, :, :],
                    classification,
                ),
                dim=1,
            )
        return Prediction(self, steps, x)

    def loss(self, image: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return the classification loss.

        :param image: Input image, BCWH.
        :param label: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        assert image.shape[1] == self.num_channels, "Tensor must be in BCWH order."
        class_channels = image[:, -self.num_output_channels :, :, :]

        # Create one-hot ground truth tensor, where all pixels of the predicted class are
        # active in the respective classification channel.
        if self.pixel_wise_loss:
            y = torch.ones((image.shape[0], image.shape[2], image.shape[3])).to(
                self.device
            )
            # if binary images are classified: mask with first image channel
            if self.num_image_channels == 1:
                mask = image[:, 0, :, :] > 0
            # mask alpha channel / designated mask channel
            else:
                mask = image[:, 3, :, :] > 0
            for i in range(image.shape[0]):
                y[i] *= label[i]
            loss_ce = (
                F.cross_entropy(
                    class_channels,
                    y.long(),
                    reduction="none",
                )
                * mask
            ).mean()
            loss_classification = loss_ce
        else:
            if self.use_classifier:
                y_pred = class_channels[:, :, 0, 0]
            else:
                y_pred = torch.mean(class_channels, dim=(2, 3))

            loss_ce = (
                F.cross_entropy(
                    y_pred,
                    label.squeeze(),
                    reduction="none",
                )
            ).mean()
            loss_classification = loss_ce

        loss = loss_classification
        return {
            "total": loss,
            "classification": loss_classification,
        }

    def metrics(self, pred: torch.Tensor, label: torch.Tensor) -> Dict[str, float]:
        """
        Return dict of standard evaluation metrics.

        :param pred [torch.Tensor]: Predicted image (BWHC).
        :param label [torch.Tensor]: Ground truth label.
        """
        accuracy_macro_metric = torchmetrics.classification.MulticlassAccuracy(
            average="macro", num_classes=self.num_classes
        ).to(self.device)
        accuracy_micro_metric = torchmetrics.classification.MulticlassAccuracy(
            average="micro", num_classes=self.num_classes
        ).to(self.device)
        auroc_metric = torchmetrics.classification.MulticlassAUROC(
            num_classes=self.num_classes
        ).to(self.device)
        f1_metric = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.num_classes
        ).to(self.device)

        assert (
            pred.shape[1] == self.num_channels
        ), "Prediction tensor must be in BCWH order"

        class_channels = pred[
            :, self.num_image_channels + self.num_hidden_channels :, :, :
        ]
        if self.use_classifier:
            y_prob = class_channels[:, :, 0, 0]
        else:
            y_prob = torch.mean(class_channels, dim=(2, 3))
        y_true = label
        if len(y_true.shape) == 2:
            y_true = label.squeeze(1)

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
        }
