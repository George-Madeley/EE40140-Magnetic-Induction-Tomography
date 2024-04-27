from torch import nn, cat

class Expandor(nn.Module):
  def __init__(self, width, height, structure):
    super().__init__()

    self.width = width
    self.height = height

    self.model = [
      nn.Sequential(
        nn.Conv2d(structure[0], structure[0], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(structure[0], structure[1], kernel_size=2, stride=2),
      )
    ]

    self.model += [
      nn.Sequential (
        nn.Conv2d(structure[i], structure[i+1], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(structure[i+1], structure[i+1], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(structure[i+1], structure[i+2], kernel_size=2, stride=2),
      ) for i in range(len(structure) - 2)
    ]

    self.model += [
      nn.Sequential(
        nn.Conv2d(structure[-2], structure[-1], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(structure[-1], structure[-1], kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(structure[-1], 1, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256, 1024),
        nn.ReLU(),
        nn.Linear(1024, 4800),
        nn.Sigmoid()
      )
    ]

    self.model = nn.ModuleList(self.model)


  def forward(self, inputTensors):
    inputTensor = inputTensors.pop()
    outputTensor = self.model[0](inputTensor)
    for layer in self.model[1:]:
      inputTensor = inputTensors.pop()
      concatTensor = cat((outputTensor, inputTensor), 1)
      outputTensor = layer(concatTensor)

    outputTensor = outputTensor.view(outputTensor.size(0), 1, self.height, self.width)
    return outputTensor
  