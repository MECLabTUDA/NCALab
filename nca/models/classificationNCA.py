import torch
from torch import nn
import torch.nn.functional as F

from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC

from .basicNCA import BasicNCAModel


class ClassificationNCAModel(BasicNCAModel):
    def __init__(
        self,
        device,
        num_image_channels: int,
        num_hidden_channels: int,
        num_classes: int,
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        immutable_image_channels: bool = True,
        learned_filters: int = 2,
        lambda_activity: float = 0.01,
        pixel_wise_loss: bool = False
    ):
        """_summary_

        Args:
            device (Pytorch device descriptor): _description_
            num_image_channels (int): _description_
            num_hidden_channels (int): _description_
            num_classes (int): _description_
            fire_rate (float, optional): _description_. Defaults to 0.5.
            hidden_size (int, optional): _description_. Defaults to 128.
            use_alive_mask (bool, optional): _description_. Defaults to False.
            immutable_image_channels (bool, optional): _description_. Defaults to True.
            learned_filters (int, optional): _description_. Defaults to 2.
            lambda_activity (float, optional): Activity loss weight, penalizing high NCA activity. Defaults to 0.
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
        )
        self.num_classes = num_classes
        self.lambda_activity = lambda_activity
        self.pixel_wise_loss = pixel_wise_loss

    def forward(self, x: torch.Tensor, steps: int = 1):
        x = super().forward(x, steps)
        return x

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

            # Average over all pixels
            y_pred = F.softmax(class_channels, dim=-1)
            y_pred = torch.mean(y_pred, 1)
            y_pred = torch.mean(y_pred, 1)

            # If reduce enabled, reduce to a single scalar.
            # Otherwise, return logits of all channels as a vector.
            if reduce:
                y_pred = torch.argmax(y_pred, dim=1)
                return y_pred
            return y_pred

    def loss(self, x, target, pixel_wise=False):
        """_summary_

        Args:
            x (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        hidden_channels = x[..., self.num_image_channels : -self.num_output_channels]
        class_channels = x[..., self.num_image_channels + self.num_hidden_channels :]

        # Create one-hot ground truth tensor, where all pixels of the predicted class are
        # active in the respective classification channel.
        if self.pixel_wise_loss:
            y = torch.ones((x.shape[0], x.shape[1], x.shape[2])).to(self.device)
            for i in range(x.shape[0]):
                y[i] *= target[i]

            loss_ce = F.cross_entropy(class_channels.transpose(3, 1), y.long())
        else:
            y_pred = F.softmax(class_channels, dim=-1)
            y_pred = torch.mean(y_pred, dim=1)
            y_pred = torch.mean(y_pred, dim=1)
            loss_ce = F.cross_entropy(y_pred, target.squeeze().long())

        # Activity loss, mildly penalizes highly active NCAs.
        # We want to enforce the NCA model to "focus" on important areas for classification,
        # so that masking away inactive pixels during inference becomes more effective.
        loss_activity = torch.sum(torch.square(hidden_channels)) / (
            x.shape[0] * x.shape[1] * x.shape[2]
        )

        loss = (
            (1 - self.lambda_activity) * loss_ce
            + self.lambda_activity * loss_activity
        )
        return loss

    def validate(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        steps: int,
        batch_iteration: int,
        summary_writer=None,
    ):
        """_summary_

        Args:
            x (torch.Tensor): _description_
            target (torch.Tensor): _description_
            steps (int): _description_
            batch_iteration (int): _description_
            summary_writer (_type_, optional): _description_. Defaults to None.
        """
        with torch.no_grad():
            y_pred = self.classify(x.to(self.device), steps, reduce=True)
            y_prob = self.classify(x.to(self.device), steps, reduce=False)

            metric = MulticlassAccuracy(average="macro", num_classes=self.num_classes)
            metric.update(y_pred, y_true.squeeze().to(self.device))
            accuracy_macro = metric.compute()

            metric = MulticlassAccuracy(average="micro", num_classes=self.num_classes)
            metric.update(y_prob, y_true.squeeze().to(self.device))
            accuracy_micro = metric.compute()

            metric = MulticlassAUROC(num_classes=self.num_classes)
            metric.update(y_prob, y_true.squeeze().to(self.device))
            auroc = metric.compute()

            if summary_writer:
                summary_writer.add_scalar(
                    "Acc/val_acc_macro", accuracy_macro, batch_iteration
                )
                summary_writer.add_scalar(
                    "Acc/val_acc_micro", accuracy_micro, batch_iteration
                )
                summary_writer.add_scalar(
                    "Acc/val_AUC", auroc, batch_iteration
                )
