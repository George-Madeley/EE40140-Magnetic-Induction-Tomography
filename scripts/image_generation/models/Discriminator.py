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
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass through the discriminator model.

        Args:
            x (Tensor): Input tensor to the discriminator.

        Returns:
            Tensor: Output tensor from the discriminator.

        """
        output = self.model(x)
        return output
