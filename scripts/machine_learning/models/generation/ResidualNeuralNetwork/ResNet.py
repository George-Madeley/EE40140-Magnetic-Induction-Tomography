from torch import nn
from torch import Tensor
from typing import List


class ResNet(nn.Module):
  """
  Residual Neural Network (ResNet) model.

  Args:
    inputSize (int): Size of the input.
    width (int): Width of the output tensor.
    height (int): Height of the output tensor.
    structure (list[int]): List specifying the structure of the model.

  Attributes:
    model (nn.ModuleList): List of model layers.
    isResidual (list[bool]): List specifying whether each layer is a residual block or not.
    width (int): Width of the output tensor.
    height (int): Height of the output tensor.
  """

  def __init__(
    self,
    inputSize: int,
    width: int,
    height: int,
    structure: List[int]
  ) -> None:
    super().__init__()

    def inLayer(outChl: int) -> nn.Sequential:
      return nn.Sequential(
        nn.Linear(inputSize, 600),
        nn.ReLU(),
        nn.BatchNorm1d(600),
        nn.Linear(600, 1200),
        nn.ReLU(),
        nn.BatchNorm1d(1200),
        nn.Unflatten(1, (1, 30, 40)),
        nn.Conv2d(1, outChl, (3, 3), padding=1),
      )

    def residualBlock(chl: int) -> nn.Sequential:
      return nn.Sequential(
        nn.ReLU(),
        nn.BatchNorm2d(chl),
        nn.Conv2d(chl, chl, (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(chl),
        nn.Conv2d(chl, chl, (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(chl),
        nn.Conv2d(chl, chl, (3, 3), padding=1)
      )

    def bottleNeck(inChl: int, botChl: int) -> nn.Sequential:
      return nn.Sequential(
        nn.ReLU(),
        nn.BatchNorm2d(inChl),
        nn.Conv2d(inChl, botChl, (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(botChl),
        nn.Conv2d(botChl, botChl, (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(botChl),
        nn.Conv2d(botChl, inChl, (3, 3), padding=1)
      )

    def doubleLayer(inChl: int, outChl: int) -> nn.Sequential:
      return nn.Sequential(
        nn.Conv2d(inChl, outChl, (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(outChl),
      )

    def avgLayer(chl: int) -> nn.Sequential:
      return nn.Sequential(
        nn.ReLU(),
        nn.BatchNorm2d(chl),
        nn.AvgPool2d((2, 2)),
      )

    def outLayer(chl: int) -> nn.Sequential:
      return nn.Sequential(
        nn.Linear(chl * 15 * 20, width * height),
        nn.Sigmoid(),
        nn.Linear(width * height, width * height),
        nn.Sigmoid(),
        nn.Linear(width * height, width * height),
        nn.Hardsigmoid()
      )

    self.model = nn.ModuleList([
      inLayer(64),
      residualBlock(64),
      residualBlock(64),
      residualBlock(64),
      doubleLayer(64, 128),
      bottleNeck(128, 32),
      avgLayer(128),
      outLayer(128)
    ])

    self.isResidual = [
      False,
      True,
      True,
      True,
      False,
      True,
      False,
      False
    ]

    self.width = width
    self.height = height

  def forward(self, x: Tensor) -> Tensor:
    """
    Forward pass of the ResNet model.

    Args:
      x (Tensor): Input tensor.

    Returns:
      Tensor: Output tensor after passing through the model.
    """
    inputTensor = x
    for idx, layer in enumerate(self.model):
      isResidual = self.isResidual[idx]

      outputTensor = layer(inputTensor)

      if isResidual:
        outputTensor = outputTensor + inputTensor

      if idx == len(self.isResidual) - 2:
        outputTensor = outputTensor.view(-1, 128 * 15 * 20)

      inputTensor = outputTensor

    outputTensor = outputTensor.view(x.size(0), 1, self.height, self.width)
    return outputTensor
