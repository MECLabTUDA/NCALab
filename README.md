# NCALab

NCALab makes it easy to create and analyze Neural Cellular Automata (NCA) implementations for various downstream tasks, such as image segmentation, classification and synthesis.


![docs](https://github.com/MECLabTUDA/NCAlab/actions/workflows/docs.yml/badge.svg)
![python-package](https://github.com/MECLabTUDA/NCAlab/actions/workflows/python-package.yml/badge.svg)
![manuscript](https://github.com/MECLabTUDA/NCAlab/actions/workflows/draft-pdf.yml/badge.svg)

![Animation of a growing lizard emoji](artwork/growing_emoji.gif)
![Animation of gastro-intestinal polyp segmentation using NCA](artwork/segmentation_kvasir_seg.gif)

## Neural Cellular Automata
NCA are a new type of neural architecture, fusing Cellular Automata and Artificial Neural Networks to create memory-efficient, robust models.
By replacing the transition function of a Cellular Automaton with a neural network model (a Multi-Layer Perceptron), they can learn from labelled input data to achieve tasks like image classification or segmentation.

![Generalized NCA Architecture](artwork/architecture.png)

Akin to a traditional cellular automaton, inference is carried out in multiple time steps (typically up to 100 steps per image).
In each time step, the cells of an image are stochastically updated by a multilayer perceptron.
Instead of a manual neighborhood aggregation (e.g. Moore or von Neumann neighborhood), neighboring cell states are determined by applying 2D depth-wise convolutions to the input image.

For a better overview on the basic NCA architecture, we recommend you to read the original 2020 [NCA Paper](https://distill.pub/2020/growing-ca/) by Mordvintsev et al.

## Features

Features of NCALab include:

  * Easy training and evaluation of NCA models
  * Cascaded multi-scale training
  * Tensorboard integration with default presets
  * Training with k-fold cross-validation
  * Convenience features: Fixing random seeds, selecting compute devices, data processing
  * Animation and visualization of NCA predictions

### Roadmap

The following features are planned for future releases of NCALab:

  * Implementation of more approaches presented in research that extend or tweak NCA models
  * Simplifyed saving and loading of trained NCA models
  * Evaluation of federated and continual learning with NCAs
  * NCAs that work on 3D voxel data

## Getting started

Perhaps the best way of getting started with NCALab is to take a look at the provided usage example tasks, starting with the Growing Emoji task.

### Usage Example Tasks

So far, the following example tasks are implemented in NCALab:

  * Image Generation:
    * Growing NCA for emoji generation
      * Training and evaluation
      * Fine-tuning of a pre-trained emoji generator
      * Hyperparameter search
  * Image Classification:
    * Self-classifying MNIST digits
    * MedMNIST image classification (PathMNIST, BloodMNIST, DermaMNIST)
  * Image Segmentation:
    * Endoscopic polyp segmentation (Kvasir-SEG, public)


You can find those example tasks inside the `tasks/` directory and its subfolders.


### Growing Lizard Example

A good starting point to get started with NCAs is the famous Growing Lizard emoji example.


```bash
uv run tasks/growing_emoji/train_growing_emoji.py
```


Run this script to generate a GIF of the trained model's prediction:

```bash
uv run tasks/growing_emoji/eval_growing_emoji.py
```

### Installation

Run

```bash
pip install ncalab
```

to install the latest release or

```bash
pip install git+https://github.com/MECLabTUDA/NCALab
```

for the most recent commit of NCALab.
We recommend to install NCALab in a virtual environment.


## Tensorboard integration

We recommend you to monitor your training progress in Tensorboard.
To launch tensorboard, run

```bash
uv run tensorboard --logdir=runs
```

in a separate terminal.
Once it is running, it should show you the URL the tensorboard server is running on, which is [localhost:6006](https://localhost:6006) by default.
Alternatively, you may use the tensorboard integration of your IDE.


# For Developers

Type checking:

```bash
uv run mypy ncalab
```

Static code analysis:

```bash
uv run ruff check ncalab
```

Testing:

```bash
uv run pytest
```


# How to Cite

Coming soon.
