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
        self.visualization_rows = ["input_image", "true_class", "pred_class"]

    def classify(self, image, steps=100):
        x = image.clone()
        x = self.forward(x, steps=steps)
        class_channels = x[..., : -self.num_output_channels]
        y_pred = torch.mean(class_channels, 1)
        y_pred = torch.mean(y_pred, 1)
        y_pred = torch.argmax(F.softmax(y_pred), axis=1)
        return y_pred

    def loss(self, x, target):
        y = torch.ones((x.shape[0], x.shape[1], x.shape[2])).to(self.device).long()
        for i in range(x.shape[0]):
            y[i] *= target[i]
        loss = F.cross_entropy(x[..., : -self.num_classes].transpose(3, 1), y.long())
        return loss
