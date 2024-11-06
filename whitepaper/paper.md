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
    affiliation: 1
  - name: Dennis Grotz
    affiliation: 1
  - name: Anirban Mukhopadhyay
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

Neural Cellular Automata (NCA) are a new family of neural network models that are employed in various imaging tasks such as image segmentation, classification and generation.
NCALab provides a generic research framework for training and evaluating NCAs, conducting hyper parameter searches and prototyping applications that build on NCAs.


# Statement of Need

NCAs are recently gaining attention in medical imaging, where they are deployed for various modalities in different downstream tasks.
In most cases, they outperform other U-Net or transformer-style architectures in terms of model size and robustness, while being on par with their accuracy.
However, there is no unified framework or reference implementation for training, evaluating and curiously experimenting with NCAs.

Research code for Neural Cellular Automata is typically organized in individual repositories for each individual research task.
Code bases follow different structures, even though the architecture is universal; in most cases, it can be defined by the number of input channels, hidden channels and output channels and the weights of the trained network.

The goal of this project is to unify the code base for many downstream task with NCAs in a shared project.
Within minutes, researchers and practitioners should be able to create a lean codebase for their task at hand, inspired by the numerous example tasks provided in the code repository.


# Target Audience



# Availability and Use

NCALab is available under the MIT License.

# Acknowledgements

This work is partially supported by Norwegian Research Council project
number 322600 Capsnetwork.

# References
