from torch import nn, flatten, exp, log
from torch.distributions import Normal

class Encoder(nn.Module):
  def __init__(self, input_dim, latent_dim):
    super(Encoder, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(input_dim, 512),
      nn.ReLU(True),
    )

    self.mean = nn.Sequential(
      nn.Linear(512, latent_dim)
    )
    self.sigma = nn.Sequential(
      nn.Linear(512, latent_dim)
    )

    self.kl = 0


  def forward(self, x):
    x = flatten(x, start_dim=1)
    x = self.model(x)
    mu = self.mean(x).to(x.device)
    sigma = exp(self.sigma(x)).to(x.device)
    normal = Normal(0, 1).sample(mu.shape).to(x.device)
    z = mu + sigma * normal
    self.kl = (sigma ** 2 + mu ** 2 - log(sigma) - 1/2).sum(1)
    return z