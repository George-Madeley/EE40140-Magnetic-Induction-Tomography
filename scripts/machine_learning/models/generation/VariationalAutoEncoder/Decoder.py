from torch import nn
import torch


class Decoder(nn.Module):
  """
  Decoder module for the Variational Autoencoder.

  Args:
    latent_dim (int): The dimension of the latent space.
    width (int): The width of the output image.
    height (int): The height of the output image.
    structure (list): A list of integers representing the structure of the decoder.

  Attributes:
    width (int): The width of the output image.
    height (int): The height of the output image.
    model (nn.Sequential): The sequential model representing the decoder.

  """

  def __init__(self, latent_dim: int, width: int, height: int, structure: list):
    super(Decoder, self).__init__()

    layers = [
      nn.Linear(latent_dim, structure[0]),
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

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the decoder.

    Args:
      x (torch.Tensor): The input tensor.

    Returns:
      torch.Tensor: The output tensor.

    """
    output = self.model(x)
    output = output.view(x.size(0), 1, self.height, self.width)
    return output
