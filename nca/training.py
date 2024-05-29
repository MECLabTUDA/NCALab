import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from .models.basicNCA import BasicNCAModel


def train_basic_nca(
    nca: BasicNCAModel,
    dataloader,
    model_path: str,
    max_iterations: int = 50000,
    gradient_clipping: bool = True,
    steps_range: tuple = (64, 96),
    save_every: int = 100,
):
    lr = 2e-3
    lr_gamma = 0.9999
    betas = (0.5, 0.5)
    optimizer = optim.Adam(nca.parameters(), lr=lr, betas=betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

    def training_iteration(x0, target, steps, optimizer, scheduler):
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
        return x, loss

    for iteration in range(max_iterations + 1):
        sample = next(iter(dataloader))
        x0, y0 = sample
        x0 = np.pad(x0, [(0, 0), (0, 19), (0, 0), (0, 0)], mode="constant")
        x0 = torch.from_numpy(x0.astype(np.float32)).to(nca.device)
        x0 = x0.transpose(1, 3)
        y0 = y0.to(nca.device)
        x, loss = training_iteration(
            x0, y0, np.random.randint(*steps_range), optimizer, scheduler
        )
        if iteration % save_every == 0:
            torch.save(nca.state_dict(), model_path)
