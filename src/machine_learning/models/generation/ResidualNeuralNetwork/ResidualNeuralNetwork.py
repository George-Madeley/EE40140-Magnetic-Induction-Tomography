import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from typing import List, Tuple

from .ResNet import ResNet
from ..IGeneration import IGeneration


class ResidualNeuralNetwork(IGeneration):
  """
  Residual Neural Network model for generation.
  """

  def __init__(self, structure: List[int], **kwargs):
    """
    Initialize the ResidualNeuralNetwork model.

    Args:
      structure (list): List specifying the structure of the ResNet.
      **kwargs: Additional keyword arguments.
    """
    super().__init__(**kwargs)

    self.model = ResNet(
      inputSize=240 if self.noise else 120,
      width=640 // self.downScaleFactor,
      height=480 // self.downScaleFactor,
      structure=structure,
    ).to(self.device)

    self.optimizer = Adam(
      self.model.parameters(),
      lr=self.learningRate,
    )

    self.rLossFunc = nn.BCELoss()

    self.modelNames: List[str] = ['ResNet']

  def train(self, trainLoader: DataLoader,
            metrics: List[str]) -> Tuple[float, dict]:
    """
    Train the ResidualNeuralNetwork model.

    Args:
      trainLoader: DataLoader for training data.
      metrics (list): List of metrics to evaluate.

    Returns:
      tuple: A tuple containing the loss and the calculated losses for each metric.
    """
    trainLoaderLen = len(trainLoader)

    for batch in trainLoader:
      realImageSamples, signalSamples, signalLabels, _ = batch
      # Create and label the real samples, the generated samples, and the
      # latent space samples. Send them to the chosen device i.e., CPU or GPU.
      realImageSamples = realImageSamples.to(device=self.device)
      signalSamples = signalSamples.to(device=self.device)

      # Zero the gradients of the optimizer.
      self.optimizer.zero_grad()

      # Generate the fake samples
      generatedImageSamples = self.model(signalSamples)

      losses = super().score(
          metrics,
          realImageSamples,
          generatedImageSamples,
          signalLabels)

      # Calculate the loss
      loss = self.rLossFunc(generatedImageSamples, realImageSamples)
      loss.backward()
      self.optimizer.step()

    losses = {f'ANN {k}': v / trainLoaderLen for k, v in losses.items()}

    return loss.item(), losses

  def test(self, testLoader: DataLoader, metrics: List[str]) -> dict:
    """
    Test the ResidualNeuralNetwork model.

    Args:
      testLoader: DataLoader for test data.
      metrics (list): List of metrics to evaluate.

    Returns:
      dict: Dictionary containing the calculated losses for each metric.
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

    losses = {f'ResNet {k}': v / testLoaderLen for k, v in losses.items()}
    return losses

  def predict(self, signalSamples) -> torch.Tensor:
    """
    Generate images using the ResidualNeuralNetwork model.

    Args:
      signalSamples: Input signal samples.

    Returns:
      torch.Tensor: Generated images.
    """
    generatedImages = self.model(signalSamples)
    generatedImages = generatedImages.detach().cpu()
    return generatedImages
