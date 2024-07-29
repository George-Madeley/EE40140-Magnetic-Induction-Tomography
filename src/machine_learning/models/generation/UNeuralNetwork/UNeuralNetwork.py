import torch
from torch import nn
from .UNet import UNet
from ..IGeneration import IGeneration
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict


class UNeuralNetwork(IGeneration):
  """
  Class representing the UNeuralNetwork model for generation.
  """

  def __init__(self, structure: List[int], **kwargs: Dict[str, int]) -> None:
    """
    Initialize the UNeuralNetwork model.

    Args:
    - structure: The structure of the UNet model.
    - kwargs: Additional keyword arguments.

    Returns:
    - None
    """
    super().__init__(**kwargs)

    self.model = UNet(
      240 if self.noise else 120,
      640 // self.downScaleFactor,
      480 // self.downScaleFactor,
      structure
    ).to(self.device)

    self.optimizerUNet = torch.optim.Adam(
      self.model.parameters(),
      lr=self.learningRate
    )

    self.uLossFunc = nn.MSELoss()

    self.modelNames = ['UNET']

  def train(self, trainLoader: DataLoader,
            metrics: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Train the UNeuralNetwork model.

    Args:
    - trainLoader: The data loader for training data.
    - metrics: The list of metrics to calculate.

    Returns:
    - tuple: A tuple containing the loss and losses dictionary.
    """
    trainLoaderLen = len(trainLoader)

    for batch in trainLoader:
      realImagesSamples, signalSamples, signalLabels, _ = batch
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      self.optimizerUNet.zero_grad()

      generatedImageSamples = self.model(signalSamples)

      loss = self.uLossFunc(realImagesSamples, generatedImageSamples)
      loss.backward()

      self.optimizerUNet.step()

      losses = super().score(
          metrics,
          realImagesSamples,
          generatedImageSamples,
          signalLabels)

    losses = {f'UNET {k}': v / trainLoaderLen for k, v in losses.items()}

    return loss.item(), losses

  def test(self, testLoader: DataLoader,
           metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Test the UNeuralNetwork model.

    Args:
    - testLoader: The data loader for test data.
    - metrics: The dictionary of metrics to calculate.

    Returns:
    - dict: The losses dictionary.
    """
    testLoaderLen = len(testLoader)

    for batch in testLoader:
      realImagesSamples, signalSamples, signalLabels, _ = batch
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      generatedImageSamples = self.model(signalSamples)

      losses = super().score(
          metrics,
          realImagesSamples,
          generatedImageSamples,
          signalLabels)

    losses = {f'UNET {k}': v / testLoaderLen for k, v in losses.items()}
    return losses

  def predict(self, signalSamples: torch.Tensor) -> torch.Tensor:
    """
    Predict the labels of the test data.

    Args:
    - signalSamples: The input signal samples.

    Returns:
    - torch.Tensor: The generated images.
    """
    generatedImages = self.model(signalSamples)
    generatedImages = generatedImages.detach().cpu()
    return generatedImages
