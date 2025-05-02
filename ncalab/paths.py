from pathlib import Path

"""
Current file directory.
"""
__CURRENT_PATH = Path(__file__).resolve().parent

"""
Project root directory.
"""
ROOT_PATH = (__CURRENT_PATH / "..").absolute()

"""
Directory in which training weights are stored.
"""
WEIGHTS_PATH = ROOT_PATH / "weights"

"""
Example task directory.
"""
TASK_PATH = ROOT_PATH / "tasks"

WEIGHTS_PATH.mkdir(exist_ok=True, parents=True)
TASK_PATH.mkdir(exist_ok=True, parents=True)
