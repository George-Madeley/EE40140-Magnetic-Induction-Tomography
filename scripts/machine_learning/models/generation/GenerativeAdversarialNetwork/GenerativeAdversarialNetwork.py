import os
import torch
from torch import nn
from torch.optim import Adam
from pandas import DataFrame

from .FeedforwardDiscriminator import FeedforwardDiscriminator
from .FeedforwardGenerator import FeedforwardGenerator
from .ConvolutionalDiscriminator import ConvolutionalDiscriminator
from .ConvolutionalGenerator import ConvolutionalGenerator
from ..IGeneration import IGeneration


class GenerativeAdversarialNetwork(IGeneration):
  def __init__(self, structure, convD: bool = False, convG: bool = False, **kwargs):
    """
    Initialize a GenerativeAdversarialNetwork object.

    Args:
        labelName (Literal['shape', 'sample'], optional): The name of the label to predict. Defaults to 'shape'.
        noise (bool, optional): Whether to add noise to the data. Defaults to False.
    """
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

  def train(
    self,
    trainLoader,
    metrics: list[str]
  ) -> None:
    """
    Train the generative adversarial network model using the provided DataFrame.

    Args:
        df (DataFrame): The input DataFrame containing the training data.

    Returns:
        None
    """

    trainLoaderLen = len(trainLoader)
    
    for batch in trainLoader:
      realImageSamples, signalSamples, signalLabels, _ = batch
      # Create and label the real samples, the generated samples, and the
      # latent space samples. Send them to the chosen device i.e., CPU or GPU.
      realImageSamples = realImageSamples.to(device=self.device)
      realImageSampleLabels = torch.ones((self.batchSize, 1)).to(device=self.device)
      signalSamples = signalSamples.to(device=self.device)
      generatedSamples = self.generator(signalSamples)
      generatedSampleLabels = torch.zeros((self.batchSize, 1)).to(device=self.device)
      allImageSamples = torch.cat((realImageSamples, generatedSamples))
      allImageSampleLabels = torch.cat((realImageSampleLabels, generatedSampleLabels))

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

      discriminatorLosses = super().score(metrics, allImageSampleLabels, outputDiscriminator, signalLabels, runPerPixelLoss=False)
      generatorLosses = super().score(metrics, realImageSamples, generatedSamples, signalLabels)

    discriminatorLosses = {f'D {k}': v / trainLoaderLen for k, v in discriminatorLosses.items()}
    generatorLosses = {f'G {k}': v / trainLoaderLen for k, v in generatorLosses.items()}

    loss = lossGenerator.item()

    losses = {
      **discriminatorLosses,
      **generatorLosses
    }

    return loss, losses

  def test(
      self,
      testLoader,
      metrics: list[str],
    ) -> float:
    """
    Test the generative adversarial network model on the given DataFrame and return the accuracy score.

    Parameters:
    - df (DataFrame): The DataFrame containing the test data.

    Returns:
    - float: The accuracy score of the generative adversarial network model on the test data.
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
      generatedSampleLabels = torch.zeros((self.batchSize, 1)).to(device=self.device)
      allSamples = torch.cat((realImageSamples, generatedSamples))
      allSampleLabels = torch.cat((realSampleLabels, generatedSampleLabels))

      # Test the discriminator
      self.discriminator.zero_grad()
      outputDiscriminator = self.discriminator(allSamples)
      discriminatorLosses = super().score(metrics, allSampleLabels, outputDiscriminator, signalLabels, runPerPixelLoss=False)

      # Test the generator
      self.generator.zero_grad()
      generatedSamples = self.generator(signalSamples)
      generatorLosses = super().score(metrics, realImageSamples, generatedSamples, signalLabels)

    discriminatorLosses = {f'D {k}': v / testLoaderLen for k, v in discriminatorLosses.items()}
    generatorLosses = {f'G {k}': v / testLoaderLen for k, v in generatorLosses.items()}

    losses = {
      **discriminatorLosses,
      **generatorLosses
    }

    return losses

  def predict(self, signalSamples) -> None:
    """
    Predict the labels of the test data

    :param predict_df: prediction dataframe

    :return: predictions
    """
    generatedImages = self.generator(signalSamples)
    generatedImages = generatedImages.detach().cpu()
    return generatedImages

  def saveModel(self, uniqueID: str) -> None:
    model_save_dir = os.path.join('models')
    os.makedirs(model_save_dir, exist_ok=True)

    save_path = os.path.join(model_save_dir, f'{self.name} G - {uniqueID}.pt')
    torch.save(self.generator.state_dict(), save_path)

    save_path = os.path.join(model_save_dir, f'{self.name} D - {uniqueID}.pt')
    torch.save(self.discriminator.state_dict(), save_path)

  def loadModel(self) -> None:
    if self.loadFile:

      for loadFile in self.loadFile:
      
        if f'{self.name} G' in loadFile:
          load_path = os.path.join('models', f'{loadFile}.pt')
          self.generator.load_state_dict(torch.load(load_path))
        elif f'{self.name} D' in loadFile:
          load_path = os.path.join('models', f'{loadFile}.pt')
          self.discriminator.load_state_dict(torch.load(load_path))
        else:
          raise FileExistsError(f'The load file provided {loadFile} does not match the model {self.name}')