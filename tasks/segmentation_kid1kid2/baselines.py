import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn
import segmentation_models_pytorch as smp

from ncalab import (
    WEIGHTS_PATH,
)

model_zoo_names = [
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "mobilenet_v2",
    "timm-mobilenetv3_small_100",
    "resnet18",
    "resnet34",
    "resnet50",
]

model_zoo_names_pretty = {
    "efficientnet-b0": "Eff.Net-B0",
    "efficientnet-b1": "Eff.Net-B1",
    "efficientnet-b2": "Eff.Net-B2",
    "mobilenet_v2": "MobileNet2",
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "timm-mobilenetv3_small_100": "MobileNet3",
}

baseline_colors = {
    "efficientnet-b0": "#14639A",
    "efficientnet-b1": "#4E8CB4",
    "efficientnet-b2": "#7EA9BD",
    "mobilenet_v2": "#C13800",
    "timm-mobilenetv3_small_100": "#D98D4B",
    "resnet18": "#158237",
    "resnet34": "#54A257",
    "resnet50": "#7FB679",
}


def make_model(name, pretrained):
    weights = "imagenet" if pretrained else None
    model = smp.Unet(
        encoder_name=name,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    if pretrained:
        for param in model.encoder.parameters():
            param.requires_grad = False
    return model


def make_model_zoo(pretrained=False) -> Dict[str, nn.Module]:
    model_zoo = {name: make_model(name, pretrained) for name in model_zoo_names}
    return model_zoo


def load_model(name, fold):
    model = make_model(name, False)
    model.load_state_dict(
        torch.load(
            WEIGHTS_PATH / f"unet_{name}_fold{fold:02d}.pth",
            weights_only=True,
        )
    )
    return model


def list_trainable_parameters(model_zoo):
    trainable_parameters = {}
    for k, v in model_zoo.items():
        model_parameters = filter(lambda p: p.requires_grad, v.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        trainable_parameters[k] = params
    return trainable_parameters


def visualize_segmentation(original_images, labels, predictions, model_name, figures_path):
    with torch.no_grad():
        # Convert tensors to numpy arrays and move to CPU
        original_images = original_images.cpu().numpy()
        if labels is not None:
            labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()

        # Number of images to display
        num_images = original_images.shape[0]

        # Create a grid of images
        fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 3))

        for i in range(num_images):
            # Original Image
            axes[i, 0].imshow(
                np.transpose(original_images[i], (1, 2, 0))
            )  # Change to HWC format
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis("off")

            # Ground Truth Mask
            if labels is not None:
                axes[i, 1].imshow(labels[i], cmap="gray")  # Assuming single channel
                axes[i, 1].set_title("Ground Truth Mask")
                axes[i, 1].axis("off")

            # Prediction Mask
            axes[i, 2].imshow(predictions[i][0], cmap="gray")  # Assuming single channel
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(figures_path / f"{model_name}.png")
        plt.close(fig)