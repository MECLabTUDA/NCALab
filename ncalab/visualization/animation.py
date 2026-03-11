from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import torch


class AnimatorStyle:
    def __init__(
        self,
        color_background,
        color_overlay,
        color_title,
        color_progress,
        underline: bool = True,
        progress_h: int = 3,
    ):
        self.color_background = color_background
        self.color_overlay = color_overlay
        self.color_title = color_title
        self.color_progress = color_progress
        self.underline = underline
        self.progress_h = progress_h

    def apply(self, fig, ax):
        plt.rcParams["axes.titlecolor"] = self.color_title
        plt.rcParams["text.antialiased"] = False
        fig.patch.set_facecolor(self.color_background)


animator_style_dark = AnimatorStyle(
    (0.1, 0.1, 0.1, 1.0),
    (0.69, 1.0, 0.69, 1.0),
    (0.9, 0.9, 0.9, 1.0),
    (1.0, 0.69, 0.16, 1.0),
)
animator_styles = {"dark": animator_style_dark}


class Animator:
    """
    Responsible for rendering NCA predictions as GIFs.
    """

    def __init__(
        self,
        nca,
        seed: torch.Tensor,
        steps: int = 100,
        interval: int = 100,
        repeat: bool = True,
        repeat_delay: int = 10000,
        overlay: bool = False,
        show_timestep: bool = True,
        hidden: bool = False,
        style: str | AnimatorStyle = "dark",
    ):
        """
        :param nca: NCA model instance
        :type nca: ncalab.BasicNCAModel
        :param seed: Input image for the NCA model
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
        fig.set_size_inches(2, 2)

        _style = style if isinstance(style, AnimatorStyle) else animator_styles[style]
        _style.apply(fig, ax)

        fpath = (
            Path(__file__) / ".." / ".." / ".." / "fonts" / "PixelOperatorMono-Bold.ttf"
        )

        # first frame is input image
        if nca.immutable_image_channels and not overlay:
            first_frame = seed[0, -nca.num_output_channels :]
        else:
            first_frame = seed[0, : nca.num_image_channels]
        first_frame_np = first_frame.permute(1, 2, 0).detach().cpu().numpy()
        first_frame_np = np.clip(first_frame_np, 0, 1)

        im = ax.imshow(
            first_frame_np,
            animated=True,
        )
        if show_timestep:
            ax.set_title("TIME STEP 0")

        images = []
        predictions = nca.record(seed, steps)
        for batch_index in range(len(seed)):
            for prediction in predictions:
                output_image = prediction.output_array[batch_index]
                output_image = output_image.transpose(1, 2, 0)
                images.append(output_image)

        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.tight_layout()

        def update(i):
            nonlocal ax, images, nca
            arr = images[i].copy()
            if not nca.immutable_image_channels:
                arr = arr[:, :, : nca.num_image_channels]
            elif hidden:
                hidden_channels = arr[
                    :,
                    :,
                    nca.num_image_channels : nca.num_image_channels
                    + nca.num_hidden_channels,
                ]
                arr = np.argmax(np.abs(hidden_channels), axis=-1)
                cmap = mpl.colormaps["Set3"]
                arr = cmap(arr.squeeze() / np.max(arr)).reshape(
                    (arr.shape[0], arr.shape[1], 4)
                )
            elif overlay:
                color = (
                    np.ones((arr.shape[0], arr.shape[1], 3)) * _style.color_overlay[:3]
                )
                A = np.clip(arr[:, :, : nca.num_image_channels], 0, 1)
                mask = np.clip(arr[:, :, -nca.num_output_channels :].squeeze(-1), 0, 1)
                alpha = 0.5
                threshold = 0.0
                A[mask > threshold] = (
                    alpha * color[mask > threshold] + (1 - alpha) * A[mask > threshold]
                )
                arr = A
            else:
                arr = arr[:, :, -nca.num_output_channels :]

            # convert to 0.0 .. 1.0 RGBA image
            arr = np.clip(arr, 0, 1)
            if arr.shape[2] == 3:
                alpha_channel = np.ones(
                    (arr.shape[0], arr.shape[1], 1), dtype=arr.dtype
                )
                arr = np.concatenate((arr, alpha_channel), axis=-1)
            # draw progress bar
            if _style.progress_h > 0:
                progress_w = int(
                    np.clip(
                        np.round(arr.shape[1] * ((i % steps) / steps)), 0, arr.shape[1]
                    )
                )
                progress_h = _style.progress_h
                arr[-progress_h:, :progress_w] = _style.color_progress
            # draw underline below title
            if _style.underline:
                arr[0, :] = _style.color_progress
            im.set_array(arr)
            # draw title
            if show_timestep:
                ax.set_title(f"TIME STEP {i % steps:3d}", font=fpath, fontsize=16)
            return (im,)

        self.animation_fig = animation.FuncAnimation(
            fig,
            update,
            frames=steps * len(seed),
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
