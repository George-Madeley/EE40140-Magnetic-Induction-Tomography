from torch import nn, cat

class UNet(nn.Module):
  def __init__(self, inputSize, width, height, structure):
    super().__init__()

    self.width = width
    self.height = height
    self.channels = [64]
    self.structure = structure

    newWidth = width
    newHeight = height
    for mpf in structure:
      if newWidth % mpf != 0 or newHeight % mpf != 0:
        raise ValueError(f"Width and height must be divisible by the downscale factor {mpf}")
      newWidth = newWidth // mpf
      newHeight = newHeight // mpf
      self.channels.append(self.channels[-1] * mpf)

    
    inLayer = [
      nn.Sequential(
        nn.Linear(inputSize, width * height),
        nn.ReLU(),
        nn.Unflatten(1, (1, height, width)),
        nn.Conv2d(1, self.channels[0], kernel_size=3, padding=1),
        nn.ReLU()
      )
    ]

    contractor = [
      nn.Sequential (
        nn.MaxPool2d(kernel_size=structure[i], stride=structure[i]),
        nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(self.channels[i+1], self.channels[i+1], kernel_size=3, padding=1),
        nn.ReLU()
      ) for i in range(0, len(structure))
    ]

    self.channels.reverse()
    structure.reverse()

    bottleneck = [
      nn.Sequential(
        nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(self.channels[0], self.channels[1], kernel_size=structure[0], stride=structure[0]),
      )
    ]

    expander = [
      nn.Sequential (
        nn.Conv2d(self.channels[i] * 2, self.channels[i], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(self.channels[i], self.channels[i], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(self.channels[i], self.channels[i+1], kernel_size=structure[i], stride=structure[i]),
      ) for i in range(1, len(structure))
    ]

    outputLayer = [
      nn.Sequential(
        nn.Conv2d(self.channels[-1] * 2, self.channels[-1], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(self.channels[-1], self.channels[-1], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(self.channels[-1], 1, kernel_size=3, padding=1),
        nn.BatchNorm2d(1),
        nn.Sigmoid(),
      )
    ]

    self.model = nn.ModuleList(inLayer + contractor + bottleneck + expander + outputLayer)


  def forward(self, x):
    inputTensor = x
    skipConnections = []
    for depth, layer in enumerate(self.model):
      outputTensor = layer(inputTensor)
      inputTensor = outputTensor

      if depth < len(self.structure):
        skipConnections.append(inputTensor)

      if 0 <= depth - len(self.channels) < len(self.structure):
        inputTensor = cat((skipConnections.pop(), inputTensor), 1)

    return outputTensor
    