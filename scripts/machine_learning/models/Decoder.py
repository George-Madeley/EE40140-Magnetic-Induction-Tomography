from torch import nn

class Decoder(nn.Module):
  def __init__(self, latent_dim, width, height):
    super(Decoder, self).__init__()
    self.width = width
    self.height = height
    self.model = nn.Sequential(
      nn.Linear(latent_dim, 512),
      nn.ReLU(True),
      nn.Linear(512, self.width * self.height),
      nn.Sigmoid()
    )

  def forward(self, x):
    output = self.model(x)
    output = output.view(x.size(0), 1, self.height, self.width)
    return output
