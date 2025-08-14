from typing import Optional

import numpy as np
import torch


class Prediction:
    """
    Stores the result of an NCA prediction, including the number of steps it took.

    Sequences are typically stored by BasicNCAModel's "record" function, and are
    returned as a list of Prediction objects.
    """

    def __init__(self, model, steps: int, output_image: torch.Tensor):
        """
        Constructor is typically not called explicitly.
        Rather, the forward pass of BasicNCAModel (and its
        subclasses) is responsible for filling its attributes.

        :param model: Reference to model used for prediction.
        :type model: ncalab.BasicNCAModel
        :param steps: Number of steps taken for the prediction.
        :type steps: int
        :param output_image: Output image tensor.
        :type output_image: torch.Tensor
        """
        self.model = model
        self.steps = steps
        assert output_image.shape[1] == model.num_channels
        self.output_image = output_image
        self._output_array: Optional[np.ndarray] = None

    @property
    def image_channels(self) -> torch.Tensor:
        """
        Convenience property to access the image channels as a Tensor.

        :returns: BCWH Tensor
        :rtype: torch.Tensor
        """
        return self.output_image[:, : self.model.num_image_channels, :, :]

    @property
    def hidden_channels(self) -> torch.Tensor:
        """
        Convenience property to access the hidden channels as a Tensor.

        :returns: BCWH Tensor
        :rtype: torch.Tensor
        """
        return self.output_image[
            :,
            self.model.num_image_channels : self.model.num_hidden_channels
            + self.model.num_hidden_channels,
            :,
            :,
        ]

    @property
    def output_channels(self) -> torch.Tensor:
        """
        Convenience property to access the output channels as a Tensor.

        :returns: BCWH Tensor
        :rtype: torch.Tensor
        """
        return self.output_image[
            :,
            -self.model.num_output_channels :,
            :,
            :,
        ]

    @property
    def output_array(self) -> np.ndarray:
        """
        Convenience property to access the whole output image in the format of
        a numpy array. Brings the entire tensor to CPU on demand, and only at
        the first call.

        :returns: Numpy array in BCWH format
        :rtype: np.ndarray
        """
        if self._output_array is None:
            self._output_array = self.output_image.detach().cpu().numpy()
        return self._output_array

    @property
    def image_channels_np(self) -> np.ndarray:
        """
        Convenience property to access the output image channels in the format of
        a numpy array. Brings the entire tensor to CPU on demand, and only at
        the first call.

        :returns: Numpy array in BCWH format
        :rtype: np.ndarray
        """
        if self._output_array is None:
            self._output_array = self.output_image.detach().cpu().numpy()
        return self._output_array[:, : self.model.num_image_channels, :, :]

    @property
    def hidden_channels_np(self) -> np.ndarray:
        """
        Convenience property to access the hidden image channels in the format of
        a numpy array. Brings the entire tensor to CPU on demand, and only at
        the first call.

        :returns: Numpy array in BCWH format
        :rtype: np.ndarray
        """
        if self._output_array is None:
            self._output_array = self.output_image.detach().cpu().numpy()
        return self._output_array[
            :,
            self.model.num_image_channels : self.model.num_hidden_channels
            + self.model.num_hidden_channels,
            :,
            :,
        ]

    @property
    def output_channels_np(self) -> np.ndarray:
        """
        Convenience property to access the image's output channels in the format of
        a numpy array. Brings the entire tensor to CPU on demand, and only at
        the first call.

        :returns: Numpy array in BCWH format
        :rtype: np.ndarray
        """
        if self._output_array is None:
            self._output_array = self.output_image.detach().cpu().numpy()
        return self._output_array[
            :,
            -self.model.num_output_channels :,
            :,
            :,
        ]
