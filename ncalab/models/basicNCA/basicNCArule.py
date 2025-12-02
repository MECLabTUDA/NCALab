import torch
from torch import nn


class BasicNCARule(nn.Module):
    """
    NCA rule module based on a two-layer Multi-Layer-Perceptron (MLP).

    :param nn: _description_
    :type nn: _type_
    """

    def __init__(
        self,
        device: torch.device,
        input_size: int,
        hidden_size: int,
        output_size: int,
        nonlinearity: type[nn.Module] = nn.ReLU,
    ):
        """
        _summary_

        :param device: _description_
        :type device: torch.device
        :param input_size: _description_
        :type input_size: int
        :param hidden_size: _description_
        :type hidden_size: int
        :param output_size: _description_
        :type output_size: int
        :param nonlinearity: _description_, defaults to nn.ReLU
        :type nonlinearity: type[nn.Module], optional
        """
        super().__init__()
        self.nonlinearity = nonlinearity
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self._build_network()
        self._initialize_network()
        self.network.to(self.device)

    def _build_network(self):
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_size,
                out_channels=self.hidden_size,
                bias=True,
                stride=1,
                padding=0,
                kernel_size=1,
            ),
            self.nonlinearity(),
            nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=self.output_size,
                bias=False,
                stride=1,
                padding=0,
                kernel_size=1,
            ),
        )

    def _initialize_network(self):
        """
        Initialize network weights of the MLP.

        We assume that the default initialization of the first layer is good enough.
        Since the final layer is purely linear and unbiased, we initalize with 0.
        """
        with torch.no_grad():
            data = self.network[-1].weight
            assert type(data) is torch.nn.parameter.Parameter
            data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: BCWH perception vector
        :type x: torch.Tensor
        :return: BCWH residual update
        :rtype: torch.Tensor
        """
        return self.network(x)

    def freeze(self, freeze_last: bool = False):
        """
        Freeze the first layer of the NCA rule network and, optionally, the final layer.

        :param freeze_last: _description_, defaults to False
        :type freeze_last: bool, optional
        """
        layers = self.network
        if not freeze_last:
            layers = self.network[:-1]
        for layer in layers:
            layer.requires_grad_(False)
