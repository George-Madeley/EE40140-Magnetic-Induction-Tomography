from torch import nn
from torch.optim import Adam
from pandas import DataFrame

from .ArtificialNeuralNetwork import ANN
from .IGeneration import IGeneration

class NeuralNetwork(IGeneration):
  def __init__(self, structure, **kwargs,):
    super().__init__(**kwargs)

    self.network = ANN(
      inputSize= 240 if self.noise else 120,
      width=640 // self.downScaleFactor,
      height=480 // self.downScaleFactor,
      structure=structure,
    ).to(self.device)

    self.optimizer = Adam(
      self.network.parameters(),
      lr=self.learningRate,
    )

    self.aLossFunc = nn.BCELoss()

    self.modelNames = ['ANN']

  def train(
    self,
    trainLoader,
  ):
    
    for realImageSamples, signalSamples in trainLoader:
      # Create and label the real samples, the generated samples, and the
      # latent space samples. Send them to the chosen device i.e., CPU or GPU.
      realImageSamples = realImageSamples.to(device=self.device)
      signalSamples = signalSamples.to(device=self.device)

      # Zero the gradients of the optimizer.
      self.optimizer.zero_grad()

      # Generate the fake samples
      generatedImageSamples = self.network(signalSamples)

      # Calculate the loss
      loss = self.aLossFunc(generatedImageSamples, realImageSamples)
      loss.backward()
      self.optimizer.step()

    return loss
  
  def test(
    self,
    testLoader,
    losses: dict,
  ):
    testLoaderLen = len(testLoader)

    for realImagesSamples, signalSamples in testLoader:
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      generatedImageSamples = self.network(signalSamples)

      losses = super().score(losses, realImagesSamples, generatedImageSamples)
    
    losses = {f'ANN {k}': v / testLoaderLen for k, v in losses.items()}
    return losses
  
  def predict(self, fixedSignalSamples, imgDir, epoch) -> None:
    fixedGeneratedImages = self.network(fixedSignalSamples)
    fixedGeneratedImages = fixedGeneratedImages.detach().cpu()
    self.plotImages(
      imgDir,
      fixedGeneratedImages,
      f'epoch_{str(epoch).zfill(3)}.png',
      subtitle=f'Epoch {epoch}'
    )