Introduction to Neural Cellular Automata
=========================================

Neural Cellular Automata (NCA) are a new type of neural network models that merge the concepts
of Cellular Automata and Artificial Neural Networks.
These models operate on a cellular grid or graph, where each cell interacts with its neighbors
according to a learned transition function over multiple time steps.
This allows NCAs to evolve and adapt their states based on local interactions,
leading to emergent behaviors that can be both interesting to study and useful in practical applications,
such as medical imaging.

In NCAs, the transition function is typically parameterized by a neural network, which learns to update
the state of each cell based on its current state and the states of its neighbors.
This learning process allows NCAs to discover patterns and behaviors that may not be easily defined by
traditional rule-based systems.
The evolution of the cellular grid occurs over discrete time steps, where each update can lead to
changes in the overall configuration.

The concept of Neural Cellular Automata was first introduced in 2020 by Mordvintsev et al. in their
online publication titled `Growing Neural Cellular Automata <https://distill.pub/2020/growing-ca/>`__,
which features an online demo of a Lizard emoji that emerges from a single black pixel, illustrating
the nature of NCAs to generate complex structures from minimal initial conditions.

Since their introduction, NCAs have garnered significant interest in the research community.
A curated collection of recent advancements and resources related to Neural Cellular Automata
can be found in our `Awesome List <https://github.com/MECLabTUDA/awesome-nca>`__.
This list includes papers, code repositories, and applications that highlight the versatility
and potential of NCAs in various fields, including computer graphics, biology, and medical imaging.
