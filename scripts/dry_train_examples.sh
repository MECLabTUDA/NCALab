#!/bin/bash
set -e

uv run tasks/class_cifar10/train_class_cifar10.py --dry --max-epochs 2
uv run tasks/growing_emoji/train_growing_emoji.py --dry --epochs 2
uv run tasks/class_medmnist/train_class_dermamnist.py --dry --max-epochs 2
uv run tasks/class_medmnist/train_class_bloodmnist.py --dry --max-epochs 2
uv run tasks/class_medmnist/train_class_pathmnist.py --dry --max-epochs 2
uv run tasks/segmentation_kvasir_seg/train_segmentation_kvasir_seg.py --dry --max-epochs 2
uv run tasks/selfclass_mnist/train_selfclass_mnist.py --dry --max-epochs 2
