import torch
from torch import device, nn

class VAE(nn.Module):

  def __init__(
    self,
    device,
    input_dim=120,
    hidden_dim=400,
    latent_dim=200,
    height=480,
    width=640,
  ):
    super(VAE, self).__init__()

    self.device = device
    self.height = height
    self.width = width

    # encoder
    self.encoder = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_dim, latent_dim),
      nn.LeakyReLU(0.2)
    )

    # latent mean and variance
    self.mean_layer = nn.Linear(latent_dim, 2)
    self.logvar_layer = nn.Linear(latent_dim, 2)

    # decoder
    self.decoder = nn.Sequential(
      nn.Linear(2, latent_dim),
      nn.LeakyReLU(0.2),
      nn.Linear(latent_dim, hidden_dim),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_dim, height * width),
      nn.Sigmoid()
    )

  def encode(self, x):
    """
    Encodes the input data using the encoder network.

    Args:
      x: Input data to be encoded.

    Returns:
      mean: Mean of the encoded data.
      logvar: Log variance of the encoded data.
    """
    x = self.encoder(x)
    mean, logvar = self.mean_layer(x), self.logvar_layer(x)
    return mean, logvar

  def reparameterization(self, mean, var):
    """
    Reparameterization trick for sampling from a Gaussian distribution.

    Args:
      mean (torch.Tensor): Mean of the Gaussian distribution.
      var (torch.Tensor): Variance of the Gaussian distribution.

    Returns:
      torch.Tensor: Sampled latent variable.

    """
    epsilon = torch.randn_like(var).to(self.device)
    z = mean + var * epsilon
    return z

  def decode(self, x):
    """
    Decodes the given input tensor `x` using the decoder network.

    Args:
      x (tensor): The input tensor to be decoded.

    Returns:
      tensor: The decoded output tensor.
    """
    output = self.decoder(x)
    output = output.view(x.size(0), 1, self.height, self.width)
    return output

  def forward(self, x):
    """
    Performs the forward pass of the Variational Autoencoder.

    Args:
      x (torch.Tensor): The input tensor.

    Returns:
      torch.Tensor: The reconstructed input tensor.
      torch.Tensor: The mean of the latent space distribution.
      torch.Tensor: The log variance of the latent space distribution.
    """
    mean, logvar = self.encode(x)
    z = self.reparameterization(mean, logvar)
    x_hat = self.decode(z)
    return x_hat, mean, logvar
