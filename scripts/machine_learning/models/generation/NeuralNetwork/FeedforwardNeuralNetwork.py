from torch import nn
from torch import Tensor


class FeedforwardNeuralNetwork(nn.Module):
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

    layers = [
        nn.Linear(inputSize, structure[0]),
        nn.ReLU(),
    ]
    for i in range(len(structure) - 1):
      layers.append(nn.Linear(structure[i], structure[i + 1]))
      layers.append(nn.ReLU())

    layers.append(nn.Linear(structure[-1], width * height))
    layers.append(nn.Sigmoid())


    self.width = width
    self.height = height

    self.model = nn.Sequential(*layers)

  def forward(self, x: Tensor) -> Tensor:
    """
    Forward pass of the ANN model.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor after passing through the model.
    """
    output = self.model(x)
    output = output.view(x.size(0), 1, self.height, self.width)
    return output
