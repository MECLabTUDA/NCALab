from pathlib import Path, PosixPath

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import matplotlib.animation as animation  # type: ignore[import-untyped]


class NCAAnimator:
    def __init__(self, nca, x, steps=100):
        fig, self.ax = plt.subplots()
        fig.set_size_inches(2, 2)
        im = self.ax.imshow(x[0], animated=True)
        self.ax.set_axis_off()

        def update(i):
            x = nca(x)
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

    def save(self, path: Path | PosixPath):
        self.animation_fig.save(path)