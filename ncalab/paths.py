from pathlib import Path


CURRENT_PATH = Path(__file__).resolve().parent
ROOT_PATH = (CURRENT_PATH / "..").absolute()
WEIGHTS_PATH = ROOT_PATH / "weights"
TASK_PATH = ROOT_PATH / "tasks"

WEIGHTS_PATH.mkdir(exist_ok=True, parents=True)
TASK_PATH.mkdir(exist_ok=True, parents=True)
