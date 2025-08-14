from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.animation as animation  # type: ignore[import-untyped]
import numpy as np
import torch


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
        overlay_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
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
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Calibri", "Arial"]
        plt.rcParams["axes.titlecolor"] = (0.2, 0.2, 0.2)

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
            ax.set_title(f"Time step {0}")

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
            arr = images[i]
            if not nca.immutable_image_channels:
                arr = arr[:, :, : nca.num_image_channels]
            elif overlay:
                color = np.ones((arr.shape[0], arr.shape[1], 3)) * overlay_color
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
            arr = np.clip(arr, 0, 1)
            im.set_array(arr)
            if show_timestep:
                ax.set_title(
                    r"$\mathbf{TIME STEP\:"
                    + f"{i % steps:3d}".replace(" ", r"\:")
                    + r"}$"
                )
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
