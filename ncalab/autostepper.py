import logging


class AutoStepper:
    """
    Helps selecting number of timesteps based on NCA activity.
    """

    def __init__(
        self,
        min_steps: int = 10,
        max_steps: int = 100,
        plateau: int = 5,
        verbose: bool = False,
        threshold: float = 1e-2,
    ):
        """
        Constructor.

        :param min_steps [int]: Minimum number of timesteps to always execute. Defaults to 10.
        :param max_steps [int]: Terminate after maximum number of steps. Defaults to 100.
        :param plateau [int]: _description_. Defaults to 5.
        :param verbose [bool]: Whether to communicate. Defaults to False.
         threshold (float, optional): _description_. Defaults to 1e-2.
        """
        assert min_steps >= 1
        assert plateau >= 1
        assert max_steps > min_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.plateau = plateau
        self.verbose = verbose
        self.threshold = threshold
        self.cooldown = 0

    def check(self, step, score):
        """
        _summary_

        :param score: _description_
        :type score: _type_
        :return: _description_
        :rtype: _type_
        """
        if score >= self.threshold:
            self.cooldown = 0
        else:
            self.cooldown += 1
        if self.cooldown >= self.plateau:
            if self.verbose:
                logging.info(f"Breaking after {step} steps.")
            return True
        return False
