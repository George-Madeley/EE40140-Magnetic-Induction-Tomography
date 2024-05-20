import os
import torch
from .Encoder import Encoder
from .Decoder import Decoder
from ..IGeneration import IGeneration
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict


class VariationalAutoencoder(IGeneration):
  """Variational Autoencoder class for generating images."""

  def __init__(self, structure: Dict[str,
               Dict[str, int]], **kwargs: int) -> None:
    """
    Initialize the VariationalAutoencoder class.

    Args:
      structure (dict): A dictionary containing the structure of the encoder and decoder.
      **kwargs: Additional keyword arguments.

    """
    super().__init__(**kwargs)

    encoderStructure = structure.get('encoder')
    decoderStructure = structure.get('decoder')

    latent_dim: int = 256
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

    self.modelNames: List[str] = ['VAE']

  def train(self, trainLoader: DataLoader,
            metrics: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Train the Variational Autoencoder model.

    Args:
      trainLoader: The data loader for training data.
      metrics: A list of metrics to evaluate the model.

    Returns:
      tuple: A tuple containing the loss and losses dictionary.

    """
    trainLoaderLen: int = len(trainLoader)

    for batch in trainLoader:
      realImagesSamples, signalSamples, signalLabels, _ = batch
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      self.optimizerEncoder.zero_grad()
      self.optimizerDecoder.zero_grad()

      latentSamples = self.encoder(signalSamples)
      generatedImageSamples = self.decoder(latentSamples)

      resolution: int = (640 // self.downScaleFactor) * \
          (480 // self.downScaleFactor)
      loss: float = (((realImagesSamples - generatedImageSamples)
                     ** 2).sum() + self.encoder.kl.sum()) / resolution
      loss.backward()
      self.optimizerEncoder.step()
      self.optimizerDecoder.step()

      losses: Dict[str, float] = super().score(
        metrics, realImagesSamples, generatedImageSamples, signalLabels)

    losses = {f'VAE {k}': v / trainLoaderLen for k, v in losses.items()}

    return loss, losses

  def test(self, testLoader: DataLoader,
           metrics: List[str]) -> Dict[str, float]:
    """
    Test the Variational Autoencoder model.

    Args:
      testLoader: The data loader for test data.
      metrics: A list of metrics to evaluate the model.

    Returns:
      dict: A dictionary containing the losses.

    """
    testLoaderLen: int = len(testLoader)

    for batch in testLoader:
      realImagesSamples, signalSamples, signalLabels, _ = batch
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      latentSamples = self.encoder(signalSamples)
      generatedImageSamples = self.decoder(latentSamples)

      losses: Dict[str, float] = super().score(
        metrics, realImagesSamples, generatedImageSamples, signalLabels)

    losses = {f'VAE {k}': v / testLoaderLen for k, v in losses.items()}
    return losses

  def predict(self, signalSamples: torch.Tensor) -> torch.Tensor:
    """
    Generate images using the Variational Autoencoder model.

    Args:
      signalSamples: The input signal samples.

    Returns:
      torch.Tensor: The generated images.

    """
    latentSamples = self.encoder(signalSamples)
    generatedImages = self.decoder(latentSamples)
    generatedImages = generatedImages.detach().cpu()
    return generatedImages

  def saveModel(self, uniqueID: str) -> None:
    """
    Save the Variational Autoencoder model.

    Args:
      uniqueID (str): A unique identifier for the saved model.

    """
    model_save_dir = os.path.join('models')
    os.makedirs(model_save_dir, exist_ok=True)

    save_path = os.path.join(model_save_dir, f'{self.name} E - {uniqueID}.pt')
    torch.save(self.encoder.state_dict(), save_path)

    save_path = os.path.join(model_save_dir, f'{self.name} D - {uniqueID}.pt')
    torch.save(self.decoder.state_dict(), save_path)

  def loadModel(self) -> None:
    """
    Load the Variational Autoencoder model.

    Raises:
      FileExistsError: If the load file provided does not match the model.

    """
    if self.loadFile:

      for loadFile in self.loadFile:

        if f'{self.name} E' in loadFile:
          load_path = os.path.join('models', f'{loadFile}.pt')
          self.encoder.load_state_dict(torch.load(load_path))
        elif f'{self.name} D' in loadFile:
          load_path = os.path.join('models', f'{loadFile}.pt')
          self.decoder.load_state_dict(torch.load(load_path))
        else:
          raise FileExistsError(
            f'The load file provided {loadFile} does not match the model {self.name}')
