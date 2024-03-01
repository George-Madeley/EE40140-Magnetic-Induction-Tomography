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
        Forward pass of the generator network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 100).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 784).

        """
        output = self.model(x)
        return output
