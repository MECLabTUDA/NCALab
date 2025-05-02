from os import PathLike
import json

from .paths import WEIGHTS_PATH


class Experiment:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = WEIGHTS_PATH / (model_name + ".pth")

    @staticmethod
    def load(path: PathLike):
        with open(path, "r") as f:
            d = json.load(f)
        model_name = d["model_name"]
        experiment = Experiment(model_name)
        return experiment
