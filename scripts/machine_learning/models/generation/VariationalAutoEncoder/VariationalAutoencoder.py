import os
from pandas import DataFrame
import torch

from .Encoder import Encoder
from .Decoder import Decoder
from ..IGeneration import IGeneration

class VariationalAutoencoder(IGeneration):
  def __init__(self, structure, **kwargs):
    super().__init__(**kwargs)

    encoderStructure = structure.get('encoder')
    decoderStructure = structure.get('decoder')

    latent_dim = 256
    self.encoder = Encoder(
      240 if self.noise else 120,
      latent_dim,
      encoderStructure
    ).to(self.device)
    self.decoder = Decoder(
      latent_dim,
      640 // self.downScaleFactor,
      480 // self.downScaleFactor,
      decoderStructure
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
    trainLoader,
    metrics
  ) -> None:
    trainLoaderLen = len(trainLoader)

    for batch in trainLoader:
      realImagesSamples, signalSamples, signalLabels, _ = batch
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

      losses = super().score(metrics, realImagesSamples, generatedImageSamples, signalLabels)

    losses = {f'VAE {k}': v / trainLoaderLen for k, v in losses.items()}

    return loss, losses
  
  def test(
    self,
    testLoader,
    metrics: list[str],
  ):
    testLoaderLen = len(testLoader)

    for batch in testLoader:
      realImagesSamples, signalSamples, signalLabels, _ = batch
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      latentSamples = self.encoder(signalSamples)
      generatedImageSamples = self.decoder(latentSamples)

      losses = super().score(metrics, realImagesSamples, generatedImageSamples, signalLabels)
    
    losses = {f'VAE {k}': v / testLoaderLen for k, v in losses.items()}
    return losses
  
  def predict(self, signalSamples) -> None:
    latentSamples = self.encoder(signalSamples)
    generatedImages = self.decoder(latentSamples)
    generatedImages = generatedImages.detach().cpu()
    return generatedImages

  def saveModel(self, uniqueID: str) -> None:
    model_save_dir = os.path.join('models')
    os.makedirs(model_save_dir, exist_ok=True)

    save_path = os.path.join(model_save_dir, f'{self.name} E - {uniqueID}.pt')
    torch.save(self.encoder.state_dict(), save_path)

    save_path = os.path.join(model_save_dir, f'{self.name} D - {uniqueID}.pt')
    torch.save(self.decoder.state_dict(), save_path)

  def loadModel(self) -> None:
    if self.loadFile:

      for loadFile in self.loadFile:
      
        if f'{self.name} E' in loadFile:
          load_path = os.path.join('models', f'{loadFile}.pt')
          self.encoder.load_state_dict(torch.load(load_path))
        elif f'{self.name} D' in loadFile:
          load_path = os.path.join('models', f'{loadFile}.pt')
          self.decoder.load_state_dict(torch.load(load_path))
        else:
          raise FileExistsError(f'The load file provided {loadFile} does not match the model {self.name}')