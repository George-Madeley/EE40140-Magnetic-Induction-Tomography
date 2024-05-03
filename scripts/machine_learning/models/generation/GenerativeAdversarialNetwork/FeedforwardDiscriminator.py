from torch import nn
from torch import Tensor


class FeedforwardDiscriminator(nn.Module):
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

  def __init__(
      self,
      width: int,
      height: int,
      structure: list[int],
    ):
    super().__init__()

    layers = [
        nn.Linear(width * height, structure[0]),
        nn.ReLU(),
    ]
    for i in range(len(structure) - 1):
      layers.append(nn.Linear(structure[i], structure[i + 1]))
      layers.append(nn.ReLU())

    layers.append(nn.Linear(structure[-1], 1))
    layers.append(nn.Sigmoid())

    self.width = width
    self.height = height
    self.model = nn.Sequential(
      *layers
    )

  def forward(self, x: Tensor) -> Tensor:
    """
    Performs forward pass through the discriminator model.

    Args:
        x (Tensor): Input tensor to the discriminator.

    Returns:
        Tensor: Output tensor from the discriminator.

    """
    x = x.view(x.size(0), self.width * self.height)
    output = self.model(x)
    return output
