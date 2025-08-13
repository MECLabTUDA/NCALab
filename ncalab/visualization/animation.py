from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.animation as animation  # type: ignore[import-untyped]
import numpy as np
import torch


class Animator:
    def __init__(
        self,
        nca,
        seed: torch.Tensor,
        steps=100,
        interval=100,
        repeat=True,
        repeat_delay=3000,
        overlay=False,
    ):
        nca.eval()

        fig, ax = plt.subplots()
        fig.set_size_inches(2, 2)

        # first frame is input image
        if nca.immutable_image_channels and not overlay:
            first_frame = seed[0, -nca.num_output_channels :]
        else:
            first_frame = seed[0, : nca.num_image_channels]
        first_frame_np = first_frame.permute(1, 2, 0).detach().cpu().numpy()
        first_frame_np = np.clip(first_frame, 0, 1)

        im = ax.imshow(
            first_frame_np,
            animated=True,
        )

        predictions = nca.record(seed, steps)
        images = []
        for prediction in predictions:
            output_image = prediction.output_array[0]
            output_image = output_image.transpose(1, 2, 0)
            images.append(output_image)

        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.tight_layout()

        def update(i):
            nonlocal images, nca
            arr = images[i]
            if not nca.immutable_image_channels:
                arr = arr[:, :, : nca.num_image_channels]
            elif overlay:
                A = np.clip(arr[:, :, : nca.num_image_channels], 0, 1)
                B = np.clip(arr[:, :, -nca.num_output_channels :].squeeze(-1), 0, 1)
                alpha = 0.8
                threshold = 0.2
                beta = 0.8
                blue = A[:, :, 2]
                blue[B > threshold] = beta * (
                    alpha * B[B > threshold] + (1 - alpha) * blue[B > threshold]
                )
                A[:, :, 2] = blue
                arr = A
            else:
                arr = arr[:, :, -nca.num_output_channels :]
            im.set_array(arr)
            return (im,)

        self.animation_fig = animation.FuncAnimation(
            fig,
            update,
            frames=steps,
            interval=interval,
            blit=True,
            repeat=repeat,
            repeat_delay=repeat_delay,
        )

    def save(self, path: str | Path):
        self.animation_fig.save(path)
