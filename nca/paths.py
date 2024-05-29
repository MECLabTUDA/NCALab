from pathlib import Path


CURRENT_PATH = Path(__file__).resolve().parent
ROOT_PATH = (CURRENT_PATH / "..").absolute()
WEIGHTS_PATH = ROOT_PATH / "weights"
TASK_PATH = ROOT_PATH / "tasks"
