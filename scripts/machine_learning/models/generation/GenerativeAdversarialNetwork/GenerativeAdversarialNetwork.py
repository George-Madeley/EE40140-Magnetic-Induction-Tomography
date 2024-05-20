import os
import torch
from torch import nn
from torch.optim import Adam
from .FeedforwardDiscriminator import FeedforwardDiscriminator
from .FeedforwardGenerator import FeedforwardGenerator
from .ConvolutionalDiscriminator import ConvolutionalDiscriminator
from .ConvolutionalGenerator import ConvolutionalGenerator
from ..IGeneration import IGeneration
from torch.utils.data import DataLoader


class GenerativeAdversarialNetwork(IGeneration):
  """
  A class representing a Generative Adversarial Network (GAN) model.

  Args:
    structure (dict): The structure of the discriminator and generator networks.
    convD (bool, optional): Whether to use a convolutional discriminator. Defaults to False.
    convG (bool, optional): Whether to use a convolutional generator. Defaults to False.
    **kwargs: Additional keyword arguments.

  Attributes:
    discriminator (nn.Module): The discriminator network.
    generator (nn.Module): The generator network.
    optimizerDiscriminator (torch.optim.Adam): The optimizer for the discriminator.
    optimizerGenerator (torch.optim.Adam): The optimizer for the generator.
    gLossFunc (nn.MSELoss): The loss function for the generator.
    dLossFunc (nn.BCELoss): The loss function for the discriminator.
    modelNames (list): The names of the models.

  """

  def __init__(
          self,
          structure: dict,
          convD: bool = False,
          convG: bool = False,
          **kwargs):
    super().__init__(**kwargs)

    discriminatorStructure = structure.get('discriminator')
    generatorStructure = structure.get('generator')

    if convD:
      Discriminator = ConvolutionalDiscriminator
    else:
      Discriminator = FeedforwardDiscriminator

    if convG:
      Generator = ConvolutionalGenerator
    else:
      Generator = FeedforwardGenerator

    self.discriminator = Discriminator(
      640 // self.downScaleFactor,
      480 // self.downScaleFactor,
      discriminatorStructure
    ).to(self.device)

    self.generator = Generator(
      240 if self.noise else 120,
      640 // self.downScaleFactor,
      480 // self.downScaleFactor,
      generatorStructure
    ).to(self.device)

    self.optimizerDiscriminator = Adam(
      self.discriminator.parameters(),
      lr=self.learningRate
    )
    self.optimizerGenerator = Adam(
      self.generator.parameters(),
      lr=self.learningRate
    )

    self.gLossFunc = nn.MSELoss()
    self.dLossFunc = nn.BCELoss()

    self.modelNames = ['D', 'G']

  def train(self, trainLoader: DataLoader, metrics: list) -> tuple:
    """
    Train the Generative Adversarial Network model.

    Args:
      trainLoader (DataLoader): The data loader for training data.
      metrics (list): The list of metrics to evaluate.

    Returns:
      tuple: A tuple containing the loss and losses.

    """
    trainLoaderLen = len(trainLoader)

    for batch in trainLoader:
      realImageSamples, signalSamples, signalLabels, _ = batch
      # Create and label the real samples, the generated samples, and the
      # latent space samples. Send them to the chosen device i.e., CPU or GPU.
      realImageSamples = realImageSamples.to(device=self.device)
      realImageSampleLabels = torch.ones(
          (self.batchSize, 1)).to(
          device=self.device)
      signalSamples = signalSamples.to(device=self.device)
      generatedSamples = self.generator(signalSamples)
      generatedSampleLabels = torch.zeros(
          (self.batchSize, 1)).to(
          device=self.device)
      allImageSamples = torch.cat((realImageSamples, generatedSamples))
      allImageSampleLabels = torch.cat(
          (realImageSampleLabels, generatedSampleLabels))

      # Training the discriminator
      self.discriminator.zero_grad()
      outputDiscriminator = self.discriminator(allImageSamples)
      lossDiscriminator = self.dLossFunc(
        outputDiscriminator, allImageSampleLabels
      )
      lossDiscriminator.backward()
      self.optimizerDiscriminator.step()

      # Training the generator
      self.generator.zero_grad()
      generatedSamples = self.generator(signalSamples)
      lossGenerator = self.gLossFunc(
        generatedSamples, realImageSamples
      )
      lossGenerator.backward()
      self.optimizerGenerator.step()

      discriminatorLosses = super().score(
          metrics,
          allImageSampleLabels,
          outputDiscriminator,
          signalLabels,
          runPerPixelLoss=False)
      generatorLosses = super().score(
          metrics,
          realImageSamples,
          generatedSamples,
          signalLabels)

    discriminatorLosses = {
        f'D {k}': v / trainLoaderLen for k,
        v in discriminatorLosses.items()}
    generatorLosses = {
        f'G {k}': v / trainLoaderLen for k,
        v in generatorLosses.items()}

    loss = lossGenerator.item()

    losses = {
      **discriminatorLosses,
      **generatorLosses
    }

    return loss, losses

  def test(self, testLoader: DataLoader, metrics: list) -> dict:
    """
    Test the Generative Adversarial Network model.

    Args:
      testLoader (DataLoader): The data loader for test data.
      metrics (list): The list of metrics to evaluate.

    Returns:
      dict: The losses.

    """
    testLoaderLen = len(testLoader)

    for batch in testLoader:
      realImageSamples, signalSamples, signalLabels, _ = batch
      # Create and label the real samples, the generated samples, and the
      # latent space samples. Send them to the chosen device i.e., CPU or GPU.
      realImageSamples = realImageSamples.to(device=self.device)
      realSampleLabels = torch.ones((self.batchSize, 1)).to(device=self.device)
      signalSamples = signalSamples.to(device=self.device)
      generatedSamples = self.generator(signalSamples)
      generatedSampleLabels = torch.zeros(
          (self.batchSize, 1)).to(
          device=self.device)
      allSamples = torch.cat((realImageSamples, generatedSamples))
      allSampleLabels = torch.cat((realSampleLabels, generatedSampleLabels))

      # Test the discriminator
      self.discriminator.zero_grad()
      outputDiscriminator = self.discriminator(allSamples)
      discriminatorLosses = super().score(
          metrics,
          allSampleLabels,
          outputDiscriminator,
          signalLabels,
          runPerPixelLoss=False)

      # Test the generator
      self.generator.zero_grad()
      generatedSamples = self.generator(signalSamples)
      generatorLosses = super().score(
          metrics,
          realImageSamples,
          generatedSamples,
          signalLabels)

    discriminatorLosses = {
        f'D {k}': v / testLoaderLen for k,
        v in discriminatorLosses.items()}
    generatorLosses = {
        f'G {k}': v / testLoaderLen for k,
        v in generatorLosses.items()}

    losses = {
      **discriminatorLosses,
      **generatorLosses
    }

    return losses

  def predict(self, signalSamples) -> torch.Tensor:
    """
    Generate images using the Generative Adversarial Network model.

    Args:
      signalSamples: The input signal samples.

    Returns:
      torch.Tensor: The generated images.

    """
    generatedImages = self.generator(signalSamples)
    generatedImages = generatedImages.detach().cpu()
    return generatedImages

  def saveModel(self, uniqueID: str):
    """
    Save the Generative Adversarial Network model.

    Args:
      uniqueID (str): The unique ID for the saved model.

    """
    model_save_dir = os.path.join('models')
    os.makedirs(model_save_dir, exist_ok=True)

    save_path = os.path.join(model_save_dir, f'{self.name} G - {uniqueID}.pt')
    torch.save(self.generator.state_dict(), save_path)

    save_path = os.path.join(model_save_dir, f'{self.name} D - {uniqueID}.pt')
    torch.save(self.discriminator.state_dict(), save_path)

  def loadModel(self):
    """
    Load the Generative Adversarial Network model.

    Raises:
      FileExistsError: If the load file provided does not match the model.

    """
    if self.loadFile:

      for loadFile in self.loadFile:

        if f'{self.name} G' in loadFile:
          load_path = os.path.join('models', f'{loadFile}.pt')
          self.generator.load_state_dict(torch.load(load_path))
        elif f'{self.name} D' in loadFile:
          load_path = os.path.join('models', f'{loadFile}.pt')
          self.discriminator.load_state_dict(torch.load(load_path))
        else:
          raise FileExistsError(
            f'The load file provided {loadFile} does not match the model {self.name}')
