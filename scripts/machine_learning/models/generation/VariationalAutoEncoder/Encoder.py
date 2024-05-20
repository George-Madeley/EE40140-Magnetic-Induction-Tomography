from torch import nn, flatten, exp, log
from torch.distributions import Normal
from typing import List
import torch


class Encoder(nn.Module):
  """
  Encoder module of the Variational Autoencoder (VAE) model.

  Args:
    input_dim (int): The dimensionality of the input data.
    latent_dim (int): The dimensionality of the latent space.
    structure (List[int]): A list specifying the structure of the encoder network.

  Attributes:
    model (nn.Sequential): The sequential model representing the encoder network.
    mean (nn.Sequential): The sequential model representing the mean layer of the encoder.
    sigma (nn.Sequential): The sequential model representing the sigma layer of the encoder.
    kl (float): The value of the Kullback-Leibler (KL) divergence.

  """

  def __init__(self, input_dim: int, latent_dim: int, structure: List[int]):
    super(Encoder, self).__init__()

    layers = [
      nn.Linear(input_dim, structure[0]),
      nn.ReLU(),
    ]
    for i in range(len(structure) - 1):
      layers.append(nn.Linear(structure[i], structure[i + 1]))
      layers.append(nn.ReLU())

    layers.append(nn.Linear(structure[-1], latent_dim))
    layers.append(nn.Sigmoid())

    self.model = nn.Sequential(*layers)

    self.mean = nn.Sequential(
      nn.Linear(512, latent_dim)
    )
    self.sigma = nn.Sequential(
      nn.Linear(512, latent_dim)
    )

    self.kl = 0.0

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the encoder.

    Args:
      x (torch.Tensor): The input tensor.

    Returns:
      torch.Tensor: The encoded latent space representation.

    """
    x = flatten(x, start_dim=1)
    x = self.model(x)
    mu = self.mean(x).to(x.device)
    sigma = exp(self.sigma(x)).to(x.device)
    normal = Normal(0, 1).sample(mu.shape).to(x.device)
    z = mu + sigma * normal
    self.kl = (sigma ** 2 + mu ** 2 - log(sigma) - 1 / 2).sum(1)
    return z
