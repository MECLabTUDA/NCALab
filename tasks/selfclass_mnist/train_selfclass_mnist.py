import sys, os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from nca.basicNCA import BasicNCAModel
from nca.training import train_basic_nca


def train_selfclass_mnist():
    device = ...
    model = BasicNCAModel(
        device,
        num_hidden_channels=1,
        num_hidden_channels=9,
        num_output_channels=10,
        fire_rate=0.5,
        hidden_size=128,
        use_alive_mask=False,
        immutable_image_channels=True,
        learned_filters=2,
    )
    train_basic_nca(model, dataloader, max_iterations=50000, batch_size=8)