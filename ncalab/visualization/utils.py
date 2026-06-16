import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from .color import Color


def string_ellipsis(label: str, max_len: int = 8, ellipsis: str = "…"):
    """
    _summary_

    :param label: _description_
    :type label: str
    :param max_len: _description_, defaults to 8
    :type max_len: int, optional
    :param ellipsis: _description_, defaults to "…"
    :type ellipsis: str, optional
    :return: _description_
    :rtype: _type_
    """
    if len(label) <= max_len:
        return label
    tokens = label.split(" ")
    if len(tokens) == 1:
        tokens = label.split("_")
    if len(tokens) > 1:
        return "".join([t[0].upper() for t in tokens])
    return label[: max_len - 1] + ellipsis


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
    normalize: bool = False,
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
    :param normalize: Whether to normalize images across batch
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Calibri", "Arial"]
    # if images are not normalized to [0, 1] or [0, 255], normalize them
    if normalize and np.min(images) < 0:
        images = (images - np.min(images)) / (np.max(images) - np.min(images))
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
                edgecolors=Color.from_rgba4b(10, 10, 10).rgba4f,
            )
        if x_index:
            ax[j].set_xlabel(r"$\mathbf{" + f"{j}" + r"}$")
        ax[j].set_xticks([])
        ax[j].set_yticks([])
        ax[j].set_aspect(1.0)
        ax[j].xaxis.label.set_color(Color.from_rgba4b(10, 10, 10).rgba4f)
        ax[j].yaxis.label.set_color(Color.from_rgba4b(10, 10, 10).rgba4f)
        [spine.set_linewidth(2) for spine in ax[j].spines.values()]
    ax[0].set_ylabel(r"$\mathbf{" + label + r"}$")
