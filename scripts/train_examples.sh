#!/bin/bash
set -e

uv run tasks/class_cifar10/train_class_cifar10.py
uv run tasks/growing_emoji/train_growing_emoji.py
uv run tasks/class_medmnist/train_class_dermamnist.py
uv run tasks/class_medmnist/train_class_bloodmnist.py
uv run tasks/class_medmnist/train_class_pathmnist.py
uv run tasks/segmentation_kvasir_seg/train_segmentation_kvasir_seg.py
uv run tasks/selfclass_mnist/train_selfclass_mnist.py
