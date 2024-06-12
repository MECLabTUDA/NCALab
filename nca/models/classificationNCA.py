import torch
import torch.nn.functional as F

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
        lambda_activity: float = 0.005,
    ):
        """_summary_

        Args:
            device (_type_): _description_
            num_image_channels (int): _description_
            num_hidden_channels (int): _description_
            num_classes (int): _description_
            fire_rate (float, optional): _description_. Defaults to 0.5.
            hidden_size (int, optional): _description_. Defaults to 128.
            use_alive_mask (bool, optional): _description_. Defaults to False.
            immutable_image_channels (bool, optional): _description_. Defaults to True.
            learned_filters (int, optional): _description_. Defaults to 2.
            lambda_activity (float, optional): Activity loss weight, penalizing high NCA activity. Defaults to 0.005.
        """
        self.num_classes = num_classes
        self.lambda_activity = lambda_activity
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

    def forward(self, x, steps: int = 1):
        x = super().forward(x, steps)
        return x

    def classify(self, image, steps: int = 100, softmax: bool = False) -> torch.Tensor:
        """_summary_

        Args:
            image (_type_): _description_
            steps (int, optional): _description_. Defaults to 100.
            softmax (bool, optional): Return vector of logits after softmax. Defaults to False.

        Returns:
            (torch.Tensor): Single class index or vector of softmax probabilities.
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

            y_pred = torch.mean(class_channels, 1)
            y_pred = torch.mean(y_pred, 1)
            if softmax:
                y_pred = torch.argmax(F.softmax(y_pred, dim=-1), axis=1)
                return y_pred
            return y_pred

    def loss(self, x, target):
        """_summary_

        Args:
            x (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        y = torch.ones((x.shape[0], x.shape[1], x.shape[2])).to(self.device).long()
        hidden_channels = x[..., self.num_image_channels : -self.num_output_channels]
        class_channels = x[..., self.num_image_channels + self.num_hidden_channels :]

        # Create one-hot ground truth tensor, where all pixels of the predicted class are
        # active in the respective classification channel.
        for i in range(x.shape[0]):
            y[i] *= target[i]
        loss_ce = F.cross_entropy(class_channels.transpose(3, 1), y.long())

        # Activity loss, mildly penalizes highly active NCAs.
        # We want to enforce the NCA model to "focus" on important areas for classification,
        # so that masking away inactive pixels during inference becomes more effective.
        loss_activity = torch.sum(torch.square(hidden_channels)) / (
            x.shape[0] * x.shape[1] * x.shape[2]
        )

        loss = (
            1 - self.lambda_activity
        ) * loss_ce + self.lambda_activity * loss_activity
        return loss
