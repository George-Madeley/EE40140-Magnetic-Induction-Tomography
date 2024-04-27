from torch import nn

class Contractor(nn.Module):
  def __init__(self, structure):
    super().__init__()

    structure = [1] + structure

    self.model = [
      nn.Sequential (
        nn.Conv2d(structure[0], structure[1], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(structure[1], structure[1], kernel_size=3, padding=1),
        nn.ReLU()
      )
    ]

    self.model += [
      nn.Sequential (
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(structure[i], structure[i+1], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(structure[i+1], structure[i+1], kernel_size=3, padding=1),
        nn.ReLU()
      ) for i in range(1, len(structure) - 1)
    ]

    self.model = nn.ModuleList(self.model)


  def forward(self, x):
    # pad the second dimension with 0s until it has a size of 256. This emulates
    # a voltage difference between a coil and itself (i.e. no voltage difference)
    maxSize = 256 if x.size(1) == 240 else 128
    while x.size(1) < maxSize:
      x = nn.functional.pad(x, (0, 1))
    
    numSignals = x.size(1)
    width = int(numSignals / 16)
    height = int(numSignals / width)

    inputTensor = x.unsqueeze(1).unsqueeze(-1)
    inputTensor = inputTensor.view(x.size(0), 1, height, width)

    skipConnections = []
    for layer in self.model:
      outputTensor = layer(inputTensor)
      skipConnections.append(outputTensor)
      inputTensor = outputTensor

    return skipConnections