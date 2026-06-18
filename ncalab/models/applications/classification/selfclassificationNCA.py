from typing import Dict, List, Literal, Optional

import torch  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]
import torchmetrics
import torchmetrics.classification

from ncalab.models.basicNCA import AbstractNCAModel
from ncalab.prediction import Prediction
from ncalab.visualization import VisualBinaryImageClassification


class SelfClassificationNCAModel(AbstractNCAModel):
    """
    Model for "self-classification", referring to "Self-Classifying MNIST Digits" by
    Mordvintsev et al.
    """

    def __init__(
        self,
        device: torch.device,
        num_hidden_channels: int,
        num_classes: int,
        fire_rate: float = 0.8,
        hidden_size: int = 128,
        num_learned_filters: int = 2,
        filter_padding: Literal["zero", "reflect", "replicate", "circular"] = "reflect",
        use_laplace: bool = False,
        kernel_size: int = 3,
        pad_noise: bool = False,
        use_temporal_encoding: bool = False,
        class_names: Optional[List[str]] = None,
        lambda_hidden: float = 0,
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
        :param num_learned_filters: Number of learned filters. If zero, use two sobel filters instead. Defaults to 2.
        :param filter_padding: Padding type to use. Might affect reliance on spatial cues. Defaults to "circular".
        :param pad_noise: Whether to pad input image tensor with noise in hidden / output channels
        """
        super(SelfClassificationNCAModel, self).__init__(
            device=device,
            num_image_channels=1,
            num_hidden_channels=num_hidden_channels,
            num_output_channels=num_classes,
            fire_rate=fire_rate,
            hidden_size=hidden_size,
            use_alive_mask=False,
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
        self.lambda_hidden = lambda_hidden
        if class_names is None:
            self.class_names = [str(i) for i in range(num_classes)]
        else:
            assert (
                len(class_names) == num_classes
            ), "Length of class names list must match number of classes"
            self.class_names = class_names
        self.plot_function = VisualBinaryImageClassification()

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

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def loss(self, pred: Prediction, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return the classification loss.

        :param pred: Prediction.
        :param label: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        image = pred.image_channels

        y = label[:, None, None].expand(
            -1,
            image.shape[2],
            image.shape[3],
        )
        mask = image[:, 0, :, :] > 0.1
        ce = F.cross_entropy(
            pred.output_channels,
            y.long(),
            reduction="none",
        )
        loss_ce = (ce * mask).sum() / (mask.sum() + 1e-8)
        loss_classification = loss_ce
        loss_hidden = torch.mean(torch.abs(pred.hidden_channels))

        loss = loss_classification + self.lambda_hidden * loss_hidden
        return {
            "total": loss,
            "classification": loss_classification.detach(),
            "hidden": loss_hidden.detach(),
        }

    def post_prediction(self, prediction: Prediction) -> Prediction:
        logits = prediction.logits
        mask = prediction.image_channels[:, 0:1, :, :]
        mask = (mask.repeat(1, self.num_classes, 1, 1) > 0.1).float()
        logits = (logits * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-8)
        prediction.logits = logits
        return prediction
