from torch import nn
from torch import Tensor


class FeedforwardGenerator(nn.Module):
  """
  Feedforward Generator model for Generative Adversarial Network.

  Args:
    inputSize (int): Size of the input tensor.
    width (int): Width of the output tensor.
    height (int): Height of the output tensor.
    structure (list): List of integers representing the structure of the generator model.

  Attributes:
    width (int): Width of the output tensor.
    height (int): Height of the output tensor.
    model (nn.Sequential): Sequential model representing the generator.
  """

  def __init__(
    self,
    inputSize: int,
    width: int,
    height: int,
    structure: list,
  ):
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
    self.model = nn.Sequential(
      *layers
    )

  def forward(self, x: Tensor) -> Tensor:
    """
    Forward pass of the Generator model.

    Args:
      x (Tensor): Input tensor.

    Returns:
      Tensor: Output tensor after passing through the model.
    """
    output = self.model(x)
    output = output.view(x.size(0), 1, self.height, self.width)
    return output
