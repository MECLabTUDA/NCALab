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
    ):
        nca.eval()

        fig, ax = plt.subplots()
        fig.set_size_inches(2, 2)

        # first frame is input image
        if nca.immutable_image_channels:
            first_frame = seed[0, -nca.num_output_channels :]
        else:
            first_frame = seed[0, : nca.num_image_channels]
        first_frame = first_frame.permute(1, 2, 0).detach().cpu().numpy()
        im = ax.imshow(
            first_frame,
            animated=True,
        )

        predictions = nca.record(seed, steps)
        images = []
        for prediction in predictions:
            if nca.immutable_image_channels:
                output_image = prediction.output_channels_np[0]
            else:
                output_image = prediction.image_channels_np[0]
            output_image = output_image.transpose(1, 2, 0)
            output_image = np.clip(output_image, 0, 1)
            images.append(output_image)

        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.tight_layout()

        def update(i):
            nonlocal images
            im.set_array(images[i])
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
