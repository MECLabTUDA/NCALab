from typing import Optional


class TrainingParameters:
    def __init__(
        self,
        gradient_clipping: bool = False,
        steps_range: tuple = (90, 110),
        lr: Optional[float] = None,
        lr_gamma: float = 0.9999,
        adam_betas: tuple = (0.9, 0.99),
        batch_repeat: int = 2,
        max_epochs: int = 200,
        optimizer_method: str = "adamw",
    ):
        assert batch_repeat >= 1
        assert steps_range[0] < steps_range[1]
        assert max_epochs > 0
        assert optimizer_method.lower() in (
            "adam",
            "adamw",
            "adagrad",
            "adafactor",
            "rmsprop",
            "sgd",
        )
        self._gradient_clipping = gradient_clipping
        self._steps_range = steps_range
        if lr is None:
            if optimizer_method.lower() == "sgd":
                self._lr = 1e-2
            elif optimizer_method.lower() in ("adam", "adamw"):
                self._lr = 16e-4
            elif optimizer_method.lower() == "rmsprop":
                self._lr = 1e-2
            elif optimizer_method.lower() == "adagrad":
                self._lr = 1e-2
            else:
                self._lr = 1e-2
        else:
            self._lr = lr
        self._lr_gamma = lr_gamma
        self._adam_betas = adam_betas

    def info(self) -> str:
        """
        Shows a markdown-formatted info string with training parameters.
        Useful for showing info on tensorboard to keep track of parameter changes.

        :returns [str]: Markdown-formatted info string.
        """
        s = "BasicNCATrainer Info\n"
        s += "-------------------\n"
        for attribute in (
            "_lr",
            "_lr_gamma",
            "_gradient_clipping",
            "_adam_betas",
            "_batch_repeat",
            "_max_epochs",
            "_optimizer_method",
        ):
            attribute_f = attribute.title().replace("_", " ")
            s += f"**{attribute_f}:** {getattr(self, attribute)}\n"
        return s
