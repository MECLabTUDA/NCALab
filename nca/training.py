from .basicNCA import BasicNCAModel

# TODO: training loop
#


def train_basic_nca(
    nca: BasicNCAModel, dataloader, max_iterations: int = 50000, batch_size: int = 8
): 
    for iteration in range(max_iterations):
        sample = next(dataloader)