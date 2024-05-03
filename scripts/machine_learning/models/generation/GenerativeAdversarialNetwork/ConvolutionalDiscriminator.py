from torch import nn
from torch import Tensor


class ConvolutionalDiscriminator(nn.Module):
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

    structure = [1] + structure + [1]

    hiddenLayers = [
      nn.Sequential(
        nn.Conv2d(structure[i], structure[i + 1], kernel_size=3, stride=1, padding=1),
        nn.ReLU()
      ) for i in range(len(structure) - 1)
    ]

    outputLayer = [
      nn.Sequential(
        nn.Flatten(),
        nn.Linear(width * height, 1),
        nn.Sigmoid()
      )
    ]

    self.width = width
    self.height = height
    self.model = nn.ModuleList(hiddenLayers + outputLayer)

  def forward(self, x: Tensor) -> Tensor:
    """
    Performs forward pass through the discriminator model.

    Args:
        x (Tensor): Input tensor to the discriminator.

    Returns:
        Tensor: Output tensor from the discriminator.

    """
    inputTensor = x
    for layer in self.model:
      outputTensor = layer(inputTensor)
      inputTensor = outputTensor
    return outputTensor
