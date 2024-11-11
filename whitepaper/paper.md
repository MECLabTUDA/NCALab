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
date: 5 November 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
#aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
#aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary


Neural Cellular Automata (NCA) are a new family of lightweight neural network models that are employed in various imaging tasks such as image segmentation, classification and generation.
These models are recently getting attention in the medical imaging community, thanks to their small size, their robustness and their overall versatility.
In terms of accuracy, they are often on-par with state-of-the art models, while being much smaller in size.
`NCALab` provides an extensible research framework for training and evaluating NCAs, conducting hyperparameter searches and prototyping applications that build on NCAs for image processing.


# Statement of Need

NCAs are recently gaining attention in medical imaging, where they are deployed for various modalities in different downstream tasks [@kalkhof2023med] [@kalkhof2023m3d].
In most cases, they outperform other U-Net or transformer-style architectures in terms of model size and robustness, while yielding similarly accurate predictions.
However, there is no unified framework or reference implementation for training, evaluating and experimentation with NCAs.

Research code for Neural Cellular Automata is typically organized in individual repositories for each individual downstream task.
Code bases follow different structures, even though the architecture is universal; in most cases, it can be defined by the number of input channels, hidden channels and output channels and the weights of the trained network.

The goal of this project is to unify the code base for various downstream tasks with NCAs in a shared project.
Within minutes, researchers and practitioners should be able to create a lean boilerplate for their task at hand, inspired by the numerous example tasks provided in this code repository.


# Ongoing Research

Currently, one journal submission and a conference paper utilizing `NCALab` are under review.


# Acknowledgements

This work is partially supported by Norwegian Research Council project
number 322600 (Capsnetwork).

# References
