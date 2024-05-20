from torch import nn
from torch import Tensor


class ConvolutionalGenerator(nn.Module):
  """
  Convolutional Generator model for generating images.

  Args:
    inputSize (int): Size of the input tensor.
    width (int): Width of the generated image.
    height (int): Height of the generated image.
    structure (list): List of integers representing the structure of the generator.

  Attributes:
    width (int): Width of the generated image.
    height (int): Height of the generated image.
    model (nn.ModuleList): List of modules representing the generator model.
  """

  def __init__(
    self,
    inputSize: int,
    width: int,
    height: int,
    structure: list,
  ):
    super().__init__()

    structure = [1] + structure

    inputLayer = [
      nn.Sequential(
        nn.Linear(inputSize, width * height),
        nn.ReLU(),
        nn.Unflatten(1, (1, height, width)),
      )
    ]

    hiddenLayers = [
      nn.Sequential(
        nn.Conv2d(structure[i], structure[i + 1], kernel_size=3, stride=1, padding=1),
        nn.ReLU()
      ) for i in range(len(structure) - 1)
    ]

    outputLayer = [
      nn.Sequential(
        nn.Conv2d(structure[-1], 1, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid()
      )
    ]

    self.width = width
    self.height = height
    self.model = nn.ModuleList(inputLayer + hiddenLayers + outputLayer)

  def forward(self, x: Tensor) -> Tensor:
    """
    Forward pass of the Generator model.

    Args:
      x (Tensor): Input tensor.

    Returns:
      Tensor: Output tensor after passing through the model.
    """
    inputTensor = x
    for layer in self.model:
      outputTensor = layer(inputTensor)
      inputTensor = outputTensor
    return outputTensor
