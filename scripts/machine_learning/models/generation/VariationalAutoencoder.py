from pandas import DataFrame
import torch

from .Encoder import Encoder
from .Decoder import Decoder
from .IGeneration import IGeneration

class VariationalAutoencoder(IGeneration):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    latent_dim = 256
    self.encoder = Encoder(
      240 if self.noise else 120,
      latent_dim
    ).to(self.device)
    self.decoder = Decoder(
      latent_dim,
      640 // self.downScaleFactor,
      480 // self.downScaleFactor 
    ).to(self.device)

    self.optimizerEncoder = torch.optim.Adam(
      self.encoder.parameters(),
      lr=self.learningRate
    )
    self.optimizerDecoder = torch.optim.Adam(
      self.decoder.parameters(),
      lr=self.learningRate
    )

    self.modelNames = ['VAE']

  def train(
    self,
    trainLoader
  ) -> None:
    for realImagesSamples, signalSamples in trainLoader:
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      self.optimizerEncoder.zero_grad()
      self.optimizerDecoder.zero_grad()

      latentSamples = self.encoder(signalSamples)
      generatedImageSamples = self.decoder(latentSamples)

      resolution = (640 // self.downScaleFactor) * (480 // self.downScaleFactor)
      loss = (((realImagesSamples - generatedImageSamples) ** 2).sum() + self.encoder.kl.sum()) / resolution
      loss.backward()
      self.optimizerEncoder.step()
      self.optimizerDecoder.step()

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

      latentSamples = self.encoder(signalSamples)
      generatedImageSamples = self.decoder(latentSamples)

      losses = super().score(losses, realImagesSamples, generatedImageSamples)
    
    losses = {f'VAE {k}': v / testLoaderLen for k, v in losses.items()}
    return losses
  
  def predict(self, fixedSignalSamples, imgDir, epoch) -> None:
    fixedLatentSamples = self.encoder(fixedSignalSamples)
    fixedGeneratedImages = self.decoder(fixedLatentSamples)
    fixedGeneratedImages = fixedGeneratedImages.detach().cpu()
    self.plotImages(imgDir, fixedGeneratedImages, f'epoch_{str(epoch).zfill(3)}.png', subtitle=f'Epoch {epoch}')
