---
title: 'NCALab: A Framework for Experimentation with Neural Cellular Automata'
tags:
  - Python
  - Neural Cellular Automata
  - Image Generation
  - Medical Image Segmentation
authors:
  - name: Henry J. Krumb
    orcid: 0000-0001-8189-4752
    corresponding: true
    affiliation: 1
  - name: Richard Sattel
    orcid: 0009-0003-1060-3462
    affiliation: 1
  - name: Dennis Grotz
    affiliation: 1
  - name: Anirban Mukhopadhyay
    orcid: 0000-0003-0669-4018
    affiliation: 1
affiliations:
 - name: Technische Universität Darmstadt, Germany
   index: 1
date: 3 April 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary


Neural Cellular Automata (NCA) are lightweight neural network models that can be employed in various image analysis tasks such as image segmentation, classification and generation.
These models are recently getting attention in the medical imaging community, thanks to their small size, their robustness and their overall versatility.
In terms of accuracy, they are often on-par with state-of-the art models, while being orders of magnitude smaller in size.
However, the training dynamics of NCAs are not yet fully understood, and there is potential for investigating practical tweaks to increase accuracy, reduce VRAM requirements and increase the overall training stability.
`NCALab` provides a unified and extensible research framework for training and evaluating NCAs, conducting hyperparameter searches and prototyping applications that build on NCAs for image processing.


# Statement of Need

NCAs are recently gaining attention in medical imaging, where they are deployed for various modalities in different downstream tasks, including 3D prostate segmentation on MRI [@kalkhof2023med] [@kalkhof2023m3d], image registration [@ranem2024ncamorph] or image synthesis [@kalkhof2024frequencytime,@kalkhof2025parameterefficient].
In most cases, they outperform other Convolutional Neural Network or Vision Transformer architectures in terms of model size and robustness, while yielding similarly accurate predictions.
However, there is no unified framework or reference implementation for training, evaluating and experimentation with NCAs.

Research code for Neural Cellular Automata is typically organized in individual repositories for each downstream task under investigation.
Code bases often follow different approaches, even though the underlying architecture is universal; in most cases, it can be defined by the number of input channels, hidden channels and output channels and the weights of the trained network.

The goal of NCALab is to provide a uniform and easy-to-use code base for various downstream tasks with NCAs in a shared project.
Within minutes, researchers and practitioners should be able to create prototypes for their ideas, inspired by the numerous example tasks provided in this code repository.
Code quality is ensured by unit tests and automated static code analysis through mypy (type checking) and flake8 (linting).

# Features

NCALab provides dedicated models and example tasks for recurring image analysis tasks, such as:

* Growing Neural Cellular Automata for emoji generation and other experiments
* Pixel-wise image segmentation
* Image classification
* Per-pixel image classification
* Monocular depth estimation

Until now, NCALab provides the following key features:

* Simplified creation, loading and training of NCA models for different downstream image analysis tasks
* Streamlined grid search for model and training hyperparameters
* Tensorboard integration to easily monitor training progress
* k-fold cross validation
* Finetuning by re-training the final layer of an NCA
* Visualization and animation of the NCA inference process


# Example use case: Training and Fine-Tuning

```python

```


# Ongoing Research

A conference paper utilizing NCALab was recently accepted for presentation in [IPCAI 2025](https://ipcai.org), and will soon be published in IJCARS.


# Acknowledgements

This work is partially supported by Norwegian Research Council project number 322600 (Capsnetwork).

# References
