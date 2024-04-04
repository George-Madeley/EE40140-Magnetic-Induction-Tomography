import torch
from torch import nn
from torch import Tensor

class Discriminator(nn.Module):
    """
    Discriminator class for image generation.

    This class defines the architecture of the discriminator model used in image
    generation. It consists of several convolutional layers followed by batch
    normalization and leaky ReLU activation. The final layer uses sigmoid
    activation to produce the discriminator output.

    Args:
        None

    Attributes:
        model (nn.Sequential):  The sequential model that defines the 
        architecture of the discriminator.

    Methods:
        forward(x: Tensor) -> Tensor:   Performs forward pass through the
        discriminator model.

    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4800, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass through the discriminator model.

        Args:
            x (Tensor): Input tensor to the discriminator.

        Returns:
            Tensor: Output tensor from the discriminator.

        """
        x = x.view(x.size(0), 4800)
        output = self.model(x)
        return output
