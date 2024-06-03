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
        fire_rate=0.5,
        hidden_size=128,
        use_alive_mask=False,
        immutable_image_channels=True,
        learned_filters=0,
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
            learned_filters (int, optional): _description_. Defaults to 0.
        """
        self.num_classes = num_classes
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

    def segment(self, image, steps=100):
        x = image.clone()
        x = self.forward(x, steps=steps)
        class_channels = x[..., : -self.num_output_channels]
        y_pred = torch.mean(class_channels, 1)
        y_pred = torch.mean(y_pred, 1)
        y_pred = torch.argmax(F.softmax(y_pred), axis=1)
        return y_pred

    def loss(self, x, target):
        loss = F.cross_entropy(x[..., : -self.num_classes].transpose(3, 1), y.long())
        return loss
