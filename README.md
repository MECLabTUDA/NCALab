# NCALab

Neural Cellular Automata (NCA) implementations for various downstream tasks, with plenty of examples, eNCApsulated in a single Python module!


![docs](https://github.com/MECLabTUDA/NCAlab/actions/workflows/docs.yml/badge.svg)
![python-package](https://github.com/MECLabTUDA/NCAlab/actions/workflows/python-package.yml/badge.svg)

![NCALab Logo](artwork/ncalab_logo.png)


## Tasks and models included in this repo

  * Growing NCA for emoji generation
  * Self-classifying MNIST digits
  * MedMNIST image classification (PathMNIST, BloodMNIST)
  * Polyp segmentation on endoscopic images (Kvasir-SEG, public)
  * Capsule endoscopic bleeding segmentation (KID2 dataset, proprietary)


## Getting started

You can find some example tasks inside the `tasks/` directory and its subfolders.

```bash
python3 tasks/growing_emoji/train_growing_emoji.py
```

```bash
python3 tasks/growing_emoji/eval_growing_emoji.py
```


## Tensorboard integration

To launch tensorboard, run

```bash
tensorboard --logdir=runs
```

in a separate terminal.
Once it is running, it should show you the URL the tensorboard server is running on, which is [localhost:6006](https://localhost:6006) by default.

Alternatively, you may use the tensorboard integration of your IDE.


## Cite this work

```bibtex
TBA
```