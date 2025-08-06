import torch


class Prediction:
    """
    Stores the result of an NCA prediction, including the number of steps it took.
    """

    def __init__(
        self, model, steps: int, output_image: torch.Tensor
    ):
        """
        Constructor. Not called explicitly; forward pass of BasicNCAModel (and its
        subclasses) is responsible for filling the attributes.

        :param model [BasicNCAModel]: Reference to model used for prediction.
        :param steps [int]: Number of steps taken for the prediction.
        :param output_image [torch.Tensor]: Output image tensor.
        """
        self.model = model
        self.steps = steps
        assert output_image.shape[1] == model.num_channels
        self.output_image = output_image

    @property
    def image_channels(self):
        """
        :returns [torch.Tensor]: BCWH
        """
        return self.output_image[:, : self.model.num_image_channels, :, :]

    @property
    def hidden_channels(self):
        """
        :returns [torch.Tensor]: BCWH
        """
        return self.output_image[
            :,
            self.model.num_image_channels : self.model.num_hidden_channels
            + self.model.num_hidden_channels,
            :,
            :,
        ]

    @property
    def output_channels(self):
        """
        :returns [torch.Tensor]: BCWH
        """
        return self.output_image[
            :,
            -self.model.num_output_channels :,
            :,
            :,
        ]
