from torch import nn

class Decoder(nn.Module):
  def __init__(self, latent_dim, width, height, structure):
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

  def forward(self, x):
    output = self.model(x)
    output = output.view(x.size(0), 1, self.height, self.width)
    return output
