#!/bin/bash
set -e

uv run tasks/class_cifar10/eval_class_cifar10.py --no-gpu
uv run tasks/growing_emoji/eval_growing_emoji.py --no-gpu
uv run tasks/class_medmnist/eval_class_dermamnist.py --no-gpu
uv run tasks/class_medmnist/eval_class_bloodmnist.py --no-gpu
uv run tasks/class_medmnist/eval_class_pathmnist.py --no-gpu
uv run tasks/segmentation_kvasir_seg/eval_segmentation_kvasir_seg.py --no-gpu
uv run tasks/selfclass_mnist/eval_selfclass_mnist.py --no-gpu

cp tasks/class_medmnist/figures/* artwork
cp tasks/class_cifar10/figures/* artwork
cp tasks/growing_emoji/figures/* artwork
cp tasks/segmentation_kvasir_seg/figures/* artwork
cp tasks/selfclass_mnist/figures/* artwork
