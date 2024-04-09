import torch
from torch import nn
from torch import Tensor

class Generator(nn.Module):
    """
    Generator class for generating images using a neural network.

    Args:
        None

    Attributes:
        model (nn.Sequential): Sequential model consisting of linear layers and activation functions.

    Methods:
        forward(x: Tensor) -> Tensor: Forward pass of the generator network.

    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(240, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 4800),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
            """
            Forward pass of the Generator model.

            Args:
                x (Tensor): Input tensor.

            Returns:
                Tensor: Output tensor after passing through the model.
            """
            output = self.model(x)
            output = output.view(x.size(0), 1, 60, 80)
            return output
