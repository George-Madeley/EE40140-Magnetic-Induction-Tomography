import torch
from torch import nn
from torch.optim import Adam
from pandas import DataFrame

from .Discriminator import Discriminator
from .Generator import Generator
from .IGeneration import IGeneration


class GenerativeAdversarialNetwork(IGeneration):
  def __init__(self, **kwargs):
    """
    Initialize a GenerativeAdversarialNetwork object.

    Args:
        labelName (Literal['shape', 'sample'], optional): The name of the label to predict. Defaults to 'shape'.
        noise (bool, optional): Whether to add noise to the data. Defaults to False.
    """
    super().__init__(**kwargs)

    self.discriminator = Discriminator(
      640 // self.downScaleFactor,
      480 // self.downScaleFactor
    ).to(self.device)
    self.generator = Generator(
      240 if self.noise else 120,
      640 // self.downScaleFactor,
      480 // self.downScaleFactor
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
  ) -> None:
    """
    Train the generative adversarial network model using the provided DataFrame.

    Args:
        df (DataFrame): The input DataFrame containing the training data.

    Returns:
        None
    """
    
    for realImageSamples, signalSamples in trainLoader:
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

    return lossDiscriminator, lossGenerator

  def test(
      self,
      testLoader,
      losses: dict,
    ) -> float:
    """
    Test the generative adversarial network model on the given DataFrame and return the accuracy score.

    Parameters:
    - df (DataFrame): The DataFrame containing the test data.

    Returns:
    - float: The accuracy score of the generative adversarial network model on the test data.
    """
    testLoaderLen = len(testLoader)

    for realSamples, latentSpaceSamples in testLoader:
      # Create and label the real samples, the generated samples, and the
      # latent space samples. Send them to the chosen device i.e., CPU or GPU.
      realSamples = realSamples.to(device=self.device)
      realSampleLabels = torch.ones((self.batchSize, 1)).to(device=self.device)
      latentSpaceSamples = latentSpaceSamples.to(device=self.device)
      generatedSamples = self.generator(latentSpaceSamples)
      generatedSampleLabels = torch.zeros((self.batchSize, 1)).to(device=self.device)
      allSamples = torch.cat((realSamples, generatedSamples))
      allSampleLabels = torch.cat((realSampleLabels, generatedSampleLabels))

      # Test the discriminator
      self.discriminator.zero_grad()
      outputDiscriminator = self.discriminator(allSamples)
      discriminatorLosses = super().score(losses, allSampleLabels, outputDiscriminator)

      # Test the generator
      self.generator.zero_grad()
      generatedSamples = self.generator(latentSpaceSamples)
      generatorLosses = super().score(losses, realSamples, generatedSamples)

    discriminatorLosses = {f'D {k}': v / testLoaderLen for k, v in discriminatorLosses.items()}
    generatorLosses = {f'G {k}': v / testLoaderLen for k, v in generatorLosses.items()}

    losses = {
      **discriminatorLosses,
      **generatorLosses
    }

    return losses

  def predict(self, fixedSignalSamples, imgDir, epoch) -> None:
    """
    Predict the labels of the test data

    :param predict_df: prediction dataframe

    :return: predictions
    """
    fixedGeneratedImages = self.generator(fixedSignalSamples)
    fixedGeneratedImages = fixedGeneratedImages.detach().cpu()
    self.plotImages(
      imgDir,
      fixedGeneratedImages,
      f'epoch_{str(epoch).zfill(3)}.png',
      subtitle=f'Epoch {epoch}'
    )
