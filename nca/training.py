import io

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

from tqdm import tqdm

from .visualization import show_batch


def train_basic_nca(
    nca,
    dataloader,
    model_path: str,
    max_iterations: int = 50000,
    gradient_clipping: bool = True,
    steps_range: tuple = (64, 96),
    save_every: int = 100,
    lr: float = 2e-3,
    lr_gamma: float = 0.9999,
    adam_betas=(0.5, 0.5),
    summary_writer=None,
    hooks: dict | None = None,
    pad_x0: bool = False
):
    def exec_hook(name, *args, **kwargs):
        if not hooks:
            return
        hook = hooks.get(name, None)
        if not hook:
            return
        return hook(*args, **kwargs)

    optimizer = optim.Adam(nca.parameters(), lr=lr, betas=adam_betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

    def training_iteration(x0, target, steps, optimizer, scheduler, batch_iteration):
        optimizer.zero_grad()
        x = x0.clone()
        x = nca(x, steps=steps)

        loss = nca.loss(x, target)
        loss.backward()

        if gradient_clipping:
            torch.nn.utils.clip_grad_norm_(nca.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if summary_writer:
            summary_writer.add_scalar("Loss/train", loss, batch_iteration)
        return x, loss

    exec_hook("pre_training", optimizer, scheduler)

    for iteration in tqdm(range(max_iterations + 1)):
        sample = next(iter(dataloader))
        x0, y0 = sample
        if pad_x0:
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
            x0 = torch.from_numpy(x0.astype(np.float32))
        x0 = x0.to(nca.device).float()
        x0 = x0.transpose(1, 3)
        y0 = y0.to(nca.device)
        x, loss = training_iteration(
            x0, y0, np.random.randint(*steps_range), optimizer, scheduler, iteration
        )
        if iteration % save_every == 0:
            exec_hook("pre_save")
            torch.save(nca.state_dict(), model_path)
            figure = show_batch(
                x0.detach().cpu().numpy(),
                x.detach().cpu().numpy(),
                y0.detach().cpu().numpy(),
                nca,
            )
            summary_writer.add_figure("Current Batch", figure, iteration)
            exec_hook("post_save")
