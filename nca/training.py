import io

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

from .models.basicNCA import BasicNCAModel


def visualize_batch(x0, x, y0):
    vis0 = x[..., 0]
    mask = x0[..., 0] > 0
    vis0 = np.array([(vis0[i] > 0) * y0[i] for i in range(vis0.shape[0])]) + 1
    vis0 *= mask
    vis0 -= 1
    vis1 = np.zeros(x.shape[0:3])
    for i in range(8):
        for j in range(28):
            for k in range(28):
                vis1[i, j, k] = np.argmax(x[i, j, k, :-10]) + 1
    vis1 *= mask
    vis1 -= 1
    figure = plt.figure(figsize=[15, 5])
    for i in range(x0.shape[0]):
        plt.subplot(2, x0.shape[0], i + 1)
        plt.imshow(vis0[i], cmap="Set3", vmin=-1, vmax=9)
        plt.axis("off")
    for i in range(x0.shape[0]):
        plt.subplot(2, x0.shape[0], i + 1 + x0.shape[0])
        plt.imshow(vis1[i], cmap="Set3", vmin=-1, vmax=9)
        plt.axis("off")
    plt.colorbar(ticks=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    return figure



def train_basic_nca(
    nca: BasicNCAModel,
    dataloader,
    model_path: str,
    max_iterations: int = 50000,
    gradient_clipping: bool = True,
    steps_range: tuple = (64, 96),
    save_every: int = 100,
    lr: float = 2e-3,
    lr_gamma: float = 0.9999,
    adam_betas=(0.5, 0.5),
    summary_writer=None
):
    optimizer = optim.Adam(nca.parameters(), lr=lr, betas=adam_betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

    def training_iteration(x0, target, steps, optimizer, scheduler, batch_iteration):
        optimizer.zero_grad()
        x = x0.clone()
        x = nca(x, steps=steps)

        y = torch.ones((x.shape[0], x.shape[1], x.shape[2])).to(nca.device).long()
        for i in range(x.shape[0]):
            y[i] *= target[i]
        loss = F.cross_entropy(x[..., :-10].transpose(3, 1), y.long())
        loss.backward()

        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(nca.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if summary_writer:
            summary_writer.add_scalar("Loss/train", loss, batch_iteration)
        return x, loss

    for iteration in range(max_iterations + 1):
        sample = next(iter(dataloader))
        x0, y0 = sample
        x0 = np.pad(
            x0,
            [
                (0, 0),  # batch
                (0, nca.num_hidden_channels + nca.num_output_channels),  # channels
                (0, 0),  # width
                (0, 0),  # height
            ],
            mode="constant",
        )
        x0 = torch.from_numpy(x0.astype(np.float32)).to(nca.device)
        x0 = x0.transpose(1, 3)
        y0 = y0.to(nca.device)
        x, loss = training_iteration(
            x0, y0, np.random.randint(*steps_range), optimizer, scheduler, iteration
        )
        if iteration % save_every == 0:
            torch.save(nca.state_dict(), model_path)
            figure = visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy(), y0.detach().cpu().numpy())
            summary_writer.add_figure("Current Batch", figure, iteration)