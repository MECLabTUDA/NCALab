from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import cv2
import matplotlib as mpl
import matplotlib.animation as animation  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import torch

from .color import Color

if TYPE_CHECKING:
    from ..models import AbstractNCAModel
from ..paths import ROOT_PATH
from ..prediction import Prediction


class AnimatorStyle:
    """ """

    def __init__(
        self,
        color_background: Color,
        color_overlay: Color,
        color_title: Color,
        color_progress: Color,
        underline: bool = True,
        progress_h: int = 3,
    ):
        """
        :param color_background: _description_
        :type color_background: Color
        :param color_overlay: Color of segmentation overlay.
        :type color_overlay: Color
        :param color_title: _description_
        :type color_title: Color
        :param color_progress: _description_
        :type color_progress: Color
        :param underline: _description_, defaults to True
        :type underline: bool, optional
        :param progress_h: _description_, defaults to 3
        :type progress_h: int, optional
        """
        self.color_background = color_background
        self.color_overlay = color_overlay
        self.color_title = color_title
        self.color_progress = color_progress
        self.underline = underline
        self.progress_h = progress_h

    def apply(self, fig, ax):
        plt.rcParams["axes.titlecolor"] = self.color_title.rgba4f
        plt.rcParams["text.antialiased"] = False
        fig.patch.set_facecolor(self.color_background.rgba4f)


animator_style_dark = AnimatorStyle(
    Color(0.1, 0.1, 0.1, 1.0),
    Color(0.69, 1.0, 0.16, 1.0),
    Color(0.9, 0.9, 0.9, 1.0),
    Color(1.0, 0.69, 0.16, 1.0),
)
animator_styles = {"dark": animator_style_dark}


def draw_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    style: AnimatorStyle,
    alpha: float = 0.5,
    contour: bool = True,
    median_denoise_kernel_size: int = 5,
) -> np.ndarray:
    """
    Helper function to draw a semi-transparent segmentation mask onto an image.

    :param image: Background image
    :type image: np.ndarray
    :param mask: Segmentation mask
    :type mask: np.ndarray
    :param style: AnimatorStyle object containing style hints
    :type style: AnimatorStyle
    :param alpha: Transparency, defaults to 0.5
    :type alpha: float, optional
    :param contour: Whether to draw a contour around the segmentation mask, defaults to True
    :type contour: bool, optional
    :param median_denoise_kernel_size: Median filter kernel size, defaults to 5
    :type median_denoise_kernel_size: int, optional
    :return: Image with segmentation overlay blended over background
    :rtype: np.ndarray
    """
    assert median_denoise_kernel_size == 0 or median_denoise_kernel_size % 2 != 0
    color = np.ones((image.shape[0], image.shape[1], 3)) * style.color_overlay.rgb3f

    if median_denoise_kernel_size > 0:
        mask = cv2.medianBlur(mask, median_denoise_kernel_size)

    # blend alpha
    image[mask > 0] = alpha * color[mask > 0] + (1 - alpha) * image[mask > 0]

    # draw contour around segmentation mask
    if contour:
        sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(sobelx, sobely)
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)[:, :]
        image[gradient_magnitude > 0] = color[gradient_magnitude > 0]
    return image


# TODO implement "input image", "hidden" and "ground truth" views that can be mixed and matched in animator.
class AnimatorView:
    def __init__(self, overlay: bool = False):
        self.overlay = overlay

    def render_frame(self, prediction: Prediction):
        pass


class AnimatorViewInput(AnimatorView):
    def __init__(self, overlay: bool = False):
        super().__init__(overlay)

    def render_frame(self, prediction: Prediction):
        pass


class Animator:
    """
    Responsible for rendering NCA predictions as GIFs.
    """

    def __init__(
        self,
        nca: "AbstractNCAModel",
        seed: torch.Tensor,
        steps: Optional[int] = None,
        interval: int = 100,
        repeat: bool = True,
        repeat_delay: int = 10000,
        overlay: bool = False,
        show_timestep: bool = True,
        hidden: bool = False,
        show_input: bool = False,
        style: str | AnimatorStyle = "dark",
    ):
        """
        :param nca: NCA model instance
        :type nca: ncalab.AbstractNCAModel
        :param seed: Input image for the NCA model, BCWH. Images in every batch are processed and their animations are concatenated.
        :type seed: torch.Tensor
        :param steps: Number of NCA prediction steps per sample, defaults to 100
        :type steps: int, optional
        :param interval: Time of each frame (milliseconds), defaults to 100
        :type interval: int, optional
        :param repeat: Whether to loop the animation, defaults to True
        :type repeat: bool, optional
        :param repeat_delay: Time after which the animation is repeated (milliseconds), defaults to 10000
        :type repeat_delay: int, optional
        :param overlay: Whether to overlay output channel (segmentation mask), defaults to False
        :type overlay: bool, optional
        :param show_timestep: Whether to display timestep in caption, defaults to True
        :type show_timestep: bool, optional
        """
        nca.eval()

        fig, ax = plt.subplots()
        w = 3 if show_input else 2
        fig.set_size_inches(w, 2)

        _style = style if isinstance(style, AnimatorStyle) else animator_styles[style]
        _style.apply(fig, ax)

        fpath = ROOT_PATH / "fonts" / "PixelOperatorMono-Bold.ttf"

        recorded_predictions: List[Prediction] = nca.record(seed, steps)
        predictions = Prediction.flatten_recorded_predictions(recorded_predictions)

        im = None

        def update(i):
            nonlocal ax, nca, im, predictions
            image = np.transpose(predictions[i].image_channels_np[0], (1, 2, 0))
            hidden_channels = np.transpose(
                predictions[i].hidden_channels_np[0], (1, 2, 0)
            )
            if predictions[i].mask is not None:
                mask = predictions[i].mask_np[0, 0]
            else:
                mask = None

            if overlay and mask is not None:
                background = np.clip(image, 0, 1)
                image = draw_segmentation_overlay(
                    background, mask.astype(np.float32), _style
                )

            arr = image

            if hidden:
                arr = np.argmax(np.abs(hidden_channels), axis=-1)
                cmap = mpl.colormaps["Set3"]
                alpha = np.clip(
                    np.max(np.abs(hidden_channels), axis=-1)
                    / np.max(np.abs(hidden_channels)),
                    0.0,
                    1.0,
                )
                arr = cmap(arr.squeeze() / np.max(arr)).reshape(
                    (arr.shape[0], arr.shape[1], 4)
                )
                arr[:, :, 3] = alpha

            # convert to 0.0 .. 1.0 RGBA image
            arr = np.clip(arr, 0, 1)
            if arr.shape[2] == 3:
                alpha_channel = np.ones(
                    (arr.shape[0], arr.shape[1], 1), dtype=arr.dtype
                )
                arr = np.concatenate((arr, alpha_channel), axis=-1)

            if hidden and show_input:
                rgb_image = image
                rgb_image -= rgb_image.min()
                rgb_image /= rgb_image.max()
                rgb_image = np.clip(rgb_image, 0, 1)
                if nca.num_image_channels == 3:
                    alpha_channel = np.ones(
                        (rgb_image.shape[0], rgb_image.shape[1], 1),
                        dtype=rgb_image.dtype,
                    )
                    rgba_image = np.concatenate((rgb_image, alpha_channel), axis=-1)
                else:
                    rgba_image = rgb_image
                arr = np.concatenate((rgba_image, arr), axis=1)

            # draw progress bar
            if _style.progress_h > 0:
                steps = len(recorded_predictions)
                progress_w = int(
                    np.clip(
                        np.round(arr.shape[1] * ((i % steps) / steps)),
                        0,
                        arr.shape[1],
                    )
                )
                progress_h = _style.progress_h
                progress_arr = np.zeros((progress_h, arr.shape[1], 4))
                progress_arr[:, :progress_w] = _style.color_progress.rgba4f
                arr = np.concatenate((arr, progress_arr), axis=0)

            # draw underline below title
            if _style.underline:
                arr[0, :] = _style.color_progress.rgba4f
            if im is None:
                im = ax.imshow(
                    arr,
                    animated=True,
                )
            im.set_array(arr)
            # draw title
            if show_timestep:
                ax.set_title(f"TIME STEP {i % steps:3d}", font=fpath, fontsize=16)
            ax.set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.tight_layout()
            return (im,)

        self.animation_fig = animation.FuncAnimation(
            fig,
            update,
            frames=len(predictions),
            interval=interval,
            blit=True,
            repeat=repeat,
            repeat_delay=repeat_delay,
        )

    def save(self, path: str | Path):
        """
        Save generated figure as GIF

        :param path: Output path
        :type path: str | Path
        """
        self.animation_fig.save(path)
