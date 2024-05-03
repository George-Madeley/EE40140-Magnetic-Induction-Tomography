import torch
from torch import nn

from .UNet import UNet
from ..IGeneration import IGeneration


class UNeuralNetwork(IGeneration):
  def __init__(self, structure, **kwargs):
    super().__init__(**kwargs)

    self.unet = UNet(
      240 if self.noise else 120,
      640 // self.downScaleFactor,
      480 // self.downScaleFactor,
      structure
    ).to(self.device)

    self.optimizerUNet = torch.optim.Adam(
      self.unet.parameters(),
      lr=self.learningRate
    )

    self.uLossFunc = nn.MSELoss()

    self.modelNames = ['UNET']

  def train(
    self,
    trainLoader,
    metrics: list[str]
  ) -> None:
    trainLoaderLen = len(trainLoader)

    for realImagesSamples, signalSamples, signalLabels in trainLoader:
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      self.optimizerUNet.zero_grad()

      generatedImageSamples = self.unet(signalSamples)

      loss = self.uLossFunc(realImagesSamples, generatedImageSamples)
      loss.backward()

      self.optimizerUNet.step()

      losses = super().score(metrics, realImagesSamples, generatedImageSamples, signalLabels)

    losses = {f'UNET {k}': v / trainLoaderLen for k, v in losses.items()}

    return loss, losses

  def test(
      self,
      testLoader,
      metrics: dict,
    ) -> float:
    """
    Test the generative adversarial network model on the given DataFrame and return the accuracy score.

    Parameters:
    - df (DataFrame): The DataFrame containing the test data.

    Returns:
    - float: The accuracy score of the generative adversarial network model on the test data.
    """
    testLoaderLen = len(testLoader)

    for realImagesSamples, signalSamples, signalLabels in testLoader:
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      generatedImageSamples = self.unet(signalSamples)

      losses = super().score(metrics, realImagesSamples, generatedImageSamples, signalLabels)
    
    losses = {f'UNET {k}': v / testLoaderLen for k, v in losses.items()}
    return losses

  def predict(self, fixedSignalSamples, imgDir, epoch) -> None:
    """
    Predict the labels of the test data

    :param predict_df: prediction dataframe

    :return: predictions
    """
    fixedGeneratedImages = self.unet(fixedSignalSamples)
    fixedGeneratedImages = fixedGeneratedImages.detach().cpu()
    self.plotImages(imgDir, fixedGeneratedImages, f'epoch_{str(epoch).zfill(3)}.png', subtitle=f'Epoch {epoch}')
