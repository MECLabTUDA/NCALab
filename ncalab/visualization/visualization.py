"""
Functions for visualizing input images and predictions
for various downstream tasks. These functions are used
for tensorboard "images" tab.
"""
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
from matplotlib.figure import Figure  # type: ignore[import-untyped]
import numpy as np

from ..prediction import Prediction


def show_image_row(
    ax,
    images,
    vmin=None,
    vmax=None,
    cmap=None,
    overlays=None,
    overlay_vmin=None,
    overlay_vmax=None,
    overlay_cmap=None,
    label: str = "",
    colorbar: bool = False,
    x_index: bool = False,
):
    """
    Shows a row of images next to each other.

    :param ax: Axis object.
    :param images: List of grayscale, RGB or RGBA images, can be CWH or WHC.
    :param vmin: Minimum value to clip channel values, defaults to None
    :param vmax: Maximum value to clip channel values, defaults to None
    :param cmap: matplotlib colormap to apply, defaults to None
    :param overlays: _description_, defaults to None
    :param overlay_vmin: _description_, defaults to None
    :param overlay_vmax: _description_, defaults to None
    :param overlay_cmap: _description_, defaults to None
    :param label: y-axis label next to first image, defaults to ""
    :param colorbar: Whether to display a colorbar next to the last image, defaults to False
    :param x_index: Whether to show the batch index below each image, defaults to False
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Calibri", "Arial"]
    for j in range(len(images)):
        image = images[j]
        # if channel dimension is first (CWH), permute to (WHC)
        if image.shape[0] in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        im = ax[j].imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
        # show colorbar next to final image if enabled
        if colorbar and j == len(images) - 1:
            plt.colorbar(im, ax=ax[j])
        if overlays is not None:
            ax[j].imshow(
                overlays[j],
                vmin=overlay_vmin,
                vmax=overlay_vmax,
                cmap=overlay_cmap,
                alpha=0.5,
                colormap="jet",
                edgecolors=(10 / 255, 10 / 255, 10 / 255),
            )
        if x_index:
            ax[j].set_xlabel(r"$\mathbf{" + f"{j}" + r"}$")
        ax[j].set_xticks([])
        ax[j].set_yticks([])
        ax[j].set_aspect(1.0)
        ax[j].xaxis.label.set_color((10 / 255, 10 / 255, 10 / 255))
        ax[j].yaxis.label.set_color((10 / 255, 10 / 255, 10 / 255))
        [spine.set_linewidth(2) for spine in ax[j].spines.values()]
    ax[0].set_ylabel(r"$\mathbf{" + label + r"}$")


class Visual:
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
        batch_size, _, image_width, image_height = image.shape

        figure, ax = plt.subplots(
            2, batch_size, figsize=[batch_size * 2, 5], tight_layout=True
        )

        # 1st row: input image
        images = (image[:, 0, :, :] > 0).astype(np.float32)
        for i in range(batch_size):
            images[i, :, :] *= label[i] + 1
        images -= 1
        show_image_row(
            ax[0], images, vmin=-1, vmax=model.num_classes, cmap="Set3", label="INPUT"
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


class VisualMultiImageClassification(Visual):
    def show(
        self, model, image: np.ndarray, prediction: Prediction, label: np.ndarray
    ) -> Figure:
        batch_size, _, image_width, image_height = image.shape

        figure, ax = plt.subplots(
            3, batch_size, figsize=[batch_size * 2, 5], tight_layout=True
        )

        # 1st row: input image
        images = np.ones((batch_size, image_width, image_height))
        hidden_channels = prediction.hidden_channels_np
        class_channels = prediction.output_channels_np
        images = prediction.image_channels_np.astype(np.float32)
        show_image_row(ax[0], images, label="IMAGE")

        for i in range(batch_size):
            mask = np.max(hidden_channels[i]) > 0.1
            class_channels[i] *= mask

        y_pred = np.mean(class_channels, (2, 3))
        y_pred = np.argmax(y_pred, axis=-1)

        # 2nd row: prediction vs. true
        for j in range(batch_size):
            ax[1, j].text(0, 1, f"TRUE: {label[j][0]}")
            ax[1, j].text(0, 0.9, f"PRED: {y_pred[j]}")
            ax[1, j].axis("off")

        # 3rd row: predicted classes per pixel
        class_channels = prediction.output_channels_np
        y_pred = np.argmax(class_channels, axis=1)
        images = (image[:, 0, :, :] > 0).astype(np.float32)
        for i in range(batch_size):
            images[i, :, :] *= y_pred[i] + 1
        images -= 1
        show_image_row(
            ax[2],
            images,
            vmin=-1,
            vmax=model.num_classes,
            cmap="Set3",
            label="PREDICTION",
        )
        figure.subplots_adjust()
        return figure


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
        show_image_row(ax[0], np.clip(images, 0.0, 1.0), label="INPUT")

        # 2nd row: true segmentation
        masks_true = label
        show_image_row(
            ax[1], np.clip(masks_true, 0.0, 1.0), cmap="gray", label="GROUND TRUTH"
        )

        # 3rd row: prediction
        masks_pred = prediction.output_channels_np
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
        show_image_row(ax[0], np.clip(images, 0.0, 1.0), label="INPUT")

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
        show_image_row(ax[0], images, label="GROUND TRUTH")

        # 2nd row: prediction
        images = prediction.image_channels_np
        show_image_row(
            ax[1], np.clip(images, 0.0, 1.0), label="PREDICTION", x_index=True
        )
        figure.subplots_adjust()
        return figure
