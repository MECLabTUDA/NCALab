import itertools
from typing import List, Optional

import numpy as np
import torch


class Prediction:
    """
    Stores the result of an NCA prediction, including the number of steps it took.

    Sequences are typically stored by BasicNCAModel's "record" function, and are
    returned as a list of Prediction objects.
    """

    def __init__(
        self,
        model,
        steps: int,
        output_image: torch.Tensor,
        logits: torch.Tensor,
        head_prediction: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
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
        self.logits = logits
        self._logits_array: Optional[np.ndarray] = None
        self.head_prediction = head_prediction
        self._head_prediction_array: Optional[np.ndarray] = None
        self.mask = mask
        self._mask_array: Optional[np.ndarray] = None

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
            self.model.num_image_channels : self.model.num_image_channels
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
            self.model.num_image_channels : self.model.num_image_channels
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

    @property
    def head_prediction_array(self) -> np.ndarray | None:
        if self.head_prediction is None:
            return None
        if self._head_prediction_array is None:
            self._head_prediction_array = self.head_prediction.detach().cpu().numpy()
        return self._head_prediction_array

    @property
    def logits_np(self) -> np.ndarray:
        if self._logits_array is None:
            self._logits_array = self.logits.detach().cpu().numpy()
        return self._logits_array

    @property
    def mask_np(self) -> np.ndarray | None:
        if self.mask is None:
            return None
        if self._mask_array is None:
            self._mask_array = self.mask.detach().cpu().numpy()
        return self._mask_array

    def unwrap_batch(self) -> List["Prediction"]:
        """
        Unwrap individual samples of a batch into a list of individual predictions.

        If your model receives an input image with batch size B, it will return
        a single Prediction instance where input image, output, logits,... all have
        batch dimensionality B. This method unwraps this single prediction into a
        list of B individual Prediction instances, each with batch size 1.

        :return: List of B Prediction instances, each with batch size 1.
        :rtype: List[Prediction]
        """
        predictions = [
            Prediction(
                self.model,
                self.steps,
                self.output_image[i : i + 1],
                logits=self.logits[i : i + 1],
                head_prediction=(
                    self.head_prediction[i : i + 1]
                    if self.head_prediction is not None
                    else None
                ),
                mask=self.mask[i : i + 1] if self.mask is not None else None,
            )
            for i in range(self.output_image.shape[0])
        ]
        return predictions

    @staticmethod
    def flatten_recorded_predictions(
        recorded_predictions: List["Prediction"],
    ) -> List["Prediction"]:
        """
        Given a list of predictions (obtained by AbstractNCAModel.record), each with
        batch size B, create a list where each individual sample (batch size 1) is chained.

        :return: List of predictions
        :rtype: List[Prediction]
        """
        if not recorded_predictions:
            return []
        unwrapped_predictions: List[List[Prediction]] = [
            p.unwrap_batch() for p in recorded_predictions
        ]
        # samples are interleaved. disentangle.
        transposed_predictions: List[List[Prediction]] = [
            list(i) for i in zip(*unwrapped_predictions)
        ]
        # flatten
        predictions_over_time: List[Prediction] = list(
            itertools.chain(*transposed_predictions)
        )
        return predictions_over_time
