"""
Functions for visualizing input images and predictions
for various downstream tasks. These functions are used
for tensorboard "images" tab.
"""

import warnings

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
from matplotlib.figure import Figure  # type: ignore[import-untyped]
from scipy.special import softmax

from .utils import string_ellipsis, show_image_row
from ..prediction import Prediction
from ..utils import unwrap


class Visual:
    """
    Base class for tensorboard visuals.
    """

    def __init__(self):
        pass

    def show(
        self, model, image: np.ndarray, prediction: Prediction, label: np.ndarray
    ) -> Figure:
        return NotImplemented


class VisualBinaryImageClassification(Visual):
    def show(
        self, model, image: np.ndarray, prediction: Prediction, label: np.ndarray
    ) -> Figure:
        batch_size, _, _, _ = image.shape

        figure, ax = plt.subplots(
            2, batch_size, figsize=[batch_size * 2, 5], tight_layout=True
        )

        # 1st row: input image
        images = (image[:, 0, :, :] > 0).astype(np.float32)
        for i in range(batch_size):
            images[i, :, :] *= label[i] + 1
        images -= 1
        show_image_row(
            ax[0],
            images,
            cmap="Set3",
            label="INPUT",
            normalize=True,
        )

        # 2nd row: prediction
        class_channels = prediction.output_channels_np
        y_pred = np.argmax(class_channels, axis=1)
        images = (image[:, 0, :, :] > 0).astype(np.float32)
        for i in range(batch_size):
            images[i, :, :] *= y_pred[i] + 1
        images -= 1
        show_image_row(
            ax[1],
            images,
            vmin=-1,
            vmax=model.num_classes,
            cmap="Set3",
            label="PREDICTION",
        )
        figure.subplots_adjust()
        return figure


class VisualRGBImageClassification(Visual):
    def show(
        self, model, image: np.ndarray, prediction: Prediction, label: np.ndarray
    ) -> Figure:
        batch_size, _, image_width, image_height = image.shape

        figure, ax = plt.subplots(
            2, batch_size, figsize=[batch_size * 2, 4], tight_layout=True
        )

        # 1st row: input image
        images = np.ones((batch_size, image_width, image_height))
        hidden_channels = prediction.hidden_channels_np
        class_channels = prediction.output_channels_np
        images = prediction.image_channels_np.astype(np.float32)
        show_image_row(ax[0], images, label="IMAGE", normalize=True)

        for i in range(batch_size):
            mask = np.max(hidden_channels[i]) > 0.1
            class_channels[i] *= mask

        if prediction.head_prediction is not None:
            y_prob = unwrap(prediction.head_prediction_array)
        else:
            y_prob = np.mean(class_channels, (2, 3))
        y_logit = softmax(y_prob, axis=-1)

        if len(label.shape) > 1:
            label = label.flatten()

        # 2nd row: prediction vs. true
        for j in range(batch_size):
            colors = [
                "xkcd:yellow green" if k == label[j] else "xkcd:cerulean"
                for k in range(len(model.class_names))
            ]
            ax[1, j].barh(
                [string_ellipsis(name) for name in model.class_names],
                y_logit[j],
                color=colors,
            )
            ax[1, j].get_xaxis().set_visible(False)
            [spine.set_linewidth(2) for spine in ax[1, j].spines.values()]
        figure.subplots_adjust()
        return figure


class VisualMultiImageClassification(Visual):
    def __init__(self):
        warnings.warn(
            "VisualMultiImageClassification is deprecated and will be removed in future versions. "
            "Please use VisualRGBImageClassification instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.new_instance = VisualRGBImageClassification()


class VisualBinaryImageSegmentation(Visual):
    def show(
        self, model, image: np.ndarray, prediction: Prediction, label: np.ndarray
    ) -> Figure:
        batch_size = image.shape[0]

        figure, ax = plt.subplots(
            3, batch_size, figsize=[batch_size * 2, 5], tight_layout=True
        )

        # 1st row: input image
        images = image[:, : model.num_image_channels, :, :]
        images = np.permute_dims(images, (0, 2, 3, 1))
        show_image_row(ax[0], np.clip(images, 0.0, 1.0), label="INPUT", normalize=True)

        # 2nd row: true segmentation
        masks_true = label
        show_image_row(
            ax[1], np.clip(masks_true, 0.0, 1.0), cmap="gray", label="GROUND TRUTH"
        )

        # 3rd row: prediction
        masks_pred = prediction.output_channels_np > 0
        show_image_row(
            ax[2], np.clip(masks_pred, 0.0, 1.0), cmap="gray", label="PREDICTION"
        )
        figure.subplots_adjust()
        return figure


class VisualDepthEstimation(Visual):
    def show(
        self, model, image: np.ndarray, prediction: Prediction, label: np.ndarray
    ) -> Figure:
        batch_size = image.shape[0]

        figure, ax = plt.subplots(
            3, batch_size, figsize=[batch_size * 2, 5], tight_layout=True
        )

        # 1st row: input image
        images = image[:, : model.num_image_channels, :, :]
        show_image_row(ax[0], np.clip(images, 0.0, 1.0), label="INPUT", normalize=True)

        # 2nd row: true segmentation
        images = label
        show_image_row(
            ax[1],
            np.clip(images, 0.0, 1.0),
            cmap="magma",
            label="GROUND TRUTH",
            colorbar=True,
        )

        # 3rd row: prediction
        images = prediction.output_channels_np
        show_image_row(
            ax[2],
            np.clip(images, 0.0, 1.0),
            cmap="magma",
            label="PREDICTION",
            colorbar=True,
        )
        figure.subplots_adjust()
        return figure


class VisualGrowing(Visual):
    def show(
        self, model, image: np.ndarray, prediction: Prediction, label: np.ndarray
    ) -> Figure:
        batch_size = image.shape[0]

        figure, ax = plt.subplots(
            2, batch_size, figsize=[batch_size * 2, 5], tight_layout=True
        )

        # 1st row: true image
        images = label[:, : model.num_image_channels, :, :]
        show_image_row(ax[0], images, label="GROUND TRUTH", normalize=True)

        # 2nd row: prediction
        images = prediction.image_channels_np
        show_image_row(
            ax[1],
            np.clip(images, 0.0, 1.0),
            label="PREDICTION",
            x_index=True,
            normalize=True,
        )
        figure.subplots_adjust()
        return figure
