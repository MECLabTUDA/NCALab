#!/bin/bash
set -e

uv run tasks/class_cifar10/eval_class_cifar10.py
uv run tasks/growing_emoji/eval_growing_emoji.py
uv run tasks/class_medmnist/eval_class_dermamnist.py
uv run tasks/class_medmnist/eval_class_bloodmnist.py
uv run tasks/class_medmnist/eval_class_pathmnist.py
uv run tasks/segmentation_kvasir_seg/eval_segmentation_kvasir_seg.py
uv run tasks/selfclass_mnist/eval_selfclass_mnist.py

cp tasks/class_medmnist/figures/* artwork
cp tasks/class_cifar10/figures/* artwork
cp tasks/growing_emoji/figures/* artwork
cp tasks/segmentation_kvasir_seg/figures/* artwork
cp tasks/selfclass_mnist/figures/* artwork
