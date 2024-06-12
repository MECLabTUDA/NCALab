import torch


def get_compute_device(device: str = "cuda:0") -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    d = torch.device(device if torch.cuda.is_available() else "cpu")
    return d
