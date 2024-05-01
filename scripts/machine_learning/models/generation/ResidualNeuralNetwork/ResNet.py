from torch import nn
from torch import Tensor


class ResNet(nn.Module):
  """
  ANN class for generating images using a neural network.

  Args:
      None

  Attributes:
      model (nn.Sequential): Sequential model consisting of linear layers and activation functions.

  Methods:
      forward(x: Tensor) -> Tensor: Forward pass of the generator network.

  """

  def __init__(self, inputSize: int, width: int, height: int, structure: list[int]):
    super().__init__()

    inLayer = lambda outChl: nn.Sequential(
      nn.Linear(inputSize, 600),
      nn.ReLU(),
      nn.BatchNorm1d(600),
      nn.Linear(600, 1200),
      nn.ReLU(),
      nn.BatchNorm1d(1200),
      nn.Unflatten(1, (1, 30, 40)),
      nn.Conv2d(1, outChl, (3, 3), padding=1),
      # nn.ReLU(),
      # nn.BatchNorm2d(1200)
    )

    residualBlock = lambda chl: nn.Sequential(
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

    bottleNeck = lambda inChl, botChl: nn.Sequential(
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

    doubleLayer = lambda inChl, outChl: nn.Sequential(
      nn.Conv2d(inChl, outChl, (3, 3), padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(outChl),
    )

    avgLayer = lambda chl: nn.Sequential(
      nn.ReLU(),
      nn.BatchNorm2d(chl),
      nn.AvgPool2d((2, 2)),
    )

    outLayer = lambda chl: nn.Sequential(
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
      residualBlock(128),
      residualBlock(128),
      doubleLayer(128, 256),
      bottleNeck(256, 64),
      residualBlock(256),
      doubleLayer(256, 512),
      residualBlock(512),
      residualBlock(512),
      residualBlock(512),
      avgLayer(512),
      outLayer(512)
    ])

    self.isResidual = [
      False,
      True,
      True,
      True,
      False,
      True,
      True,
      True,
      False,
      True,
      True,
      False,
      True,
      True,
      True,
      False,
      False
    ]

    self.width = width
    self.height = height

  def forward(self, x: Tensor) -> Tensor:
    """
    Forward pass of the ANN model.

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

      if idx == 15:
        # flatten the outputTensor from (512, 15, 20) to (512 * 15 * 20)
        outputTensor = outputTensor.view(-1, 512 * 15 * 20)

      inputTensor = outputTensor
    return outputTensor
