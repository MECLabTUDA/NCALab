from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.animation as animation  # type: ignore[import-untyped]


class NCAAnimator:
    def __init__(self, nca, x, steps=100):
        """ """
        fig, ax = plt.subplots()
        fig.set_size_inches(2, 2)
        im = ax.imshow(x[0, :3].permute(0, 2, 3, 1).cpu(), animated=True)
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.tight_layout()

        def update(i):
            nonlocal x
            x = nca(x)  # --> BWHC
            im.set_array(x)
            return (im,)

        self.animation_fig = animation.FuncAnimation(
            fig,
            update,
            frames=steps,
            interval=100,
            blit=True,
            repeat=True,
            repeat_delay=3000,
        )

    def save(self, path: str | Path):
        self.animation_fig.save(path)
