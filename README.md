# nca-models

Training code and pre-trained models for Neural Cellular Automata on different datasets and tasks.
Since NCAs are so small (less than 1 MB each), we include pre-trained weights for all tasks in this repo.


## Tasks and models included in this repo

  * Growing NCA for emoji generation
  * Self-classifying MNIST digits
  * Self-classifying MedMNIST images (PathMNIST)


## Future tasks

Tasks we want to include in the future:

  * Other MedMNIST tasks
  * FashionMNIST
  * Detection tasks

Feel free to open an issue if there is a particular task or dataset you're interested in.
However, we will only provide weights for public data.


## Getting started

You can find some example tasks inside the `tasks/` directory and its subfolders.


## Tensorboard integration

To launch tensorboard, run

```bash
tensorboard --logdir=runs
```

in a separate terminal.
Once it is running, it should show you the URL the tensorboard server is running on, which is [localhost:6006](https://localhost:6006) by default.

Alternatively, you may use the tensorboard integration of your IDE.


## Cite this work

