from torch import nn
from torch import Tensor
from typing import List


class ConvolutionalNeuralNetwork(nn.Module):
  """
  Convolutional Neural Network model.

  Args:
    inputSize (int): Size of the input.
    width (int): Width of the input image.
    height (int): Height of the input image.
    structure (List[int]): List of integers representing the structure of the network.

  Attributes:
    width (int): Width of the input image.
    height (int): Height of the input image.
    model (nn.ModuleList): List of layers in the model.
  """

  def __init__(
    self,
    inputSize: int,
    width: int,
    height: int,
    structure: List[int]
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
    Forward pass of the Convolutional Neural Network model.

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
