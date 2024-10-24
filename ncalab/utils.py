import torch
import torch.nn.functional as F


def get_compute_device(device: str = "cuda:0") -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    d = torch.device(device if torch.cuda.is_available() else "cpu")
    return d


def pad_input(x, nca, noise=True):
    if x.shape[1] < nca.num_channels:
        x = F.pad(
            x, (0, 0, 0, 0, 0, nca.num_channels - x.shape[1], 0, 0), mode="constant"
        )
        if noise:
            x[
                :,
                nca.num_image_channels : nca.num_image_channels
                + nca.num_hidden_channels,
                :,
                :,
            ] = torch.normal(
                0.5, 0.225, size=(x.shape[0], nca.num_hidden_channels, x.shape[2], x.shape[3])
            )
    return x


def NCALab_banner():
    banner = """
 _   _  _____          _           _     
| \ | |/ ____|   /\   | |         | |    
|  \| | |       /  \  | |     __ _| |__  
| . ` | |      / /\ \ | |    / _` | '_ \ 
| |\  | |____ / ____ \| |___| (_| | |_) |
|_| \_|\_____/_/    \_\______\__,_|_.__/ 
-----------------------------------------
    Developed at MECLab - TU Darmstadt
-----------------------------------------
    """
    print(banner)