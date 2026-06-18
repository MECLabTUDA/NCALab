from typing import Dict, List, Literal, Optional

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]
import torchmetrics
import torchmetrics.classification

from ncalab.losses import FocalLoss
from ncalab.models.basicNCA import AbstractNCAModel
from ncalab.prediction import Prediction
from ncalab.visualization import (
    VisualBinaryImageClassification,
    VisualRGBImageClassification,
)

from .classificationNCAhead import ClassificationNCAHead


class ClassificationNCAModel(AbstractNCAModel):
    def __init__(
        self,
        device: torch.device,
        num_image_channels: int,
        num_hidden_channels: int,
        num_classes: int,
        fire_rate: float = 0.8,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        num_learned_filters: int = 2,
        filter_padding: Literal["zero", "reflect", "replicate", "circular"] = "reflect",
        use_laplace: bool = False,
        kernel_size: int = 3,
        pad_noise: bool = False,
        use_temporal_encoding: bool = False,
        use_classifier: bool = True,
        class_names: Optional[List[str]] = None,
        avg_pool_size: int = 8,
        lambda_hidden: float = 0,
        **kwargs,
    ):
        """
        :param device: Pytorch device descriptor.
        :param num_image_channels: Number of input image channels
        :param num_hidden_channels: Number of latent channels
        :param num_classes: Number of output classes
        :param fire_rate: Fire rate for stochastic weight update. Defaults to 0.8.
        :param hidden_size: Number of neurons in hidden layer. Defaults to 128.
        :param use_alive_mask: Whether to use alive masking (channel 3) during training. Defaults to False.
        :param num_learned_filters: Number of learned filters. If zero, use two sobel filters instead. Defaults to 2.
        :param filter_padding: Padding type to use. Might affect reliance on spatial cues. Defaults to "circular".
        :param pad_noise: Whether to pad input image tensor with noise in hidden / output channels
        """
        super(ClassificationNCAModel, self).__init__(
            device=device,
            num_image_channels=num_image_channels,
            num_hidden_channels=num_hidden_channels,
            num_output_channels=0 if use_classifier else num_classes,
            fire_rate=fire_rate,
            hidden_size=hidden_size,
            use_alive_mask=use_alive_mask,
            immutable_image_channels=True,
            plot_function=None,
            num_learned_filters=num_learned_filters,
            validation_metric="f1",
            filter_padding=filter_padding,
            use_laplace=use_laplace,
            kernel_size=kernel_size,
            pad_noise=pad_noise,
            use_temporal_encoding=use_temporal_encoding,
            **kwargs,
        )
        self._num_classes = num_classes
        self.use_classifier = use_classifier
        assert avg_pool_size >= 1
        self.avg_pool_size = avg_pool_size
        self.lambda_hidden = lambda_hidden
        if use_classifier:
            self.head = ClassificationNCAHead(
                self.num_hidden_channels,
                self._num_classes,
                self.device,
                avg_pool_size=avg_pool_size,
            )
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
        self.metrics = {
            "accuracy_macro": accuracy_macro_metric,
            "accuracy_micro": accuracy_micro_metric,
            "auroc": auroc_metric,
            "f1": f1_metric,
        }
        self.focal_loss = FocalLoss()

    @property
    def num_classes(self) -> int:
        return self._num_classes

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
        prediction: Prediction = self.predict(image, steps=steps)
        if prediction.head_prediction is not None:
            class_channels = prediction.head_prediction
        else:
            class_channels = prediction.output_channels

        # if binary classification,
        # mask away pixels with the binary image used as a mask
        if self.num_image_channels == 1:
            mask = image[:, 0:1, :, :]
            class_channels *= mask

        y_pred = F.softmax(class_channels, dim=1)
        if not self.use_classifier:
            y_pred = torch.mean(y_pred, dim=(2, 3))

        # If reduce enabled, reduce to a single scalar.
        # Otherwise, return logits of all channels as a vector.
        if reduce:
            y_pred = torch.argmax(y_pred, dim=1)
            return y_pred
        return y_pred

    def loss(self, pred: Prediction, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return the classification loss.

        :param pred: Prediction.
        :param label: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        loss_focal = self.focal_loss(
            pred.logits,
            label.view(-1),
        )
        loss_classification = loss_focal
        loss_hidden = torch.mean(torch.abs(pred.hidden_channels))

        loss = loss_classification + self.lambda_hidden * loss_hidden
        return {
            "total": loss,
            "classification": loss_classification.detach(),
            "hidden": loss_hidden.detach(),
        }

    def post_prediction(self, prediction: Prediction) -> Prediction:
        logits = prediction.logits
        if not self.use_classifier:
            logits = torch.mean(logits, dim=(2, 3))
        prediction.logits = logits
        return prediction
