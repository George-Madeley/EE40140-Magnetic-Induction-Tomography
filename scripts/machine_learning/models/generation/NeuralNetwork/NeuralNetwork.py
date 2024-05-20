import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from typing import List, Tuple, Dict

from .FeedforwardNeuralNetwork import FeedforwardNeuralNetwork
from .ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from ..IGeneration import IGeneration


class NeuralNetwork(IGeneration):
  """Neural Network model for generation."""

  def __init__(self, structure: List[int], conv: bool = False, **kwargs):
    """
    Initialize the NeuralNetwork model.

    Args:
      structure (list): List specifying the structure of the neural network.
      conv (bool, optional): Flag indicating whether to use Convolutional Neural Network. Defaults to False.
      **kwargs: Additional keyword arguments.

    """
    super().__init__(**kwargs)

    if conv:
      ArtificialNeuralNetwork = ConvolutionalNeuralNetwork
    else:
      ArtificialNeuralNetwork = FeedforwardNeuralNetwork

    self.model = ArtificialNeuralNetwork(
      inputSize=240 if self.noise else 120,
      width=640 // self.downScaleFactor,
      height=480 // self.downScaleFactor,
      structure=structure,
    ).to(self.device)

    self.optimizer = Adam(
      self.model.parameters(),
      lr=self.learningRate,
    )

    self.aLossFunc = nn.BCELoss()

    self.modelNames: List[str] = ['N']

  def train(self, trainLoader: DataLoader,
            metrics: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Train the NeuralNetwork model.

    Args:
      trainLoader: DataLoader for training data.
      metrics (list): List of metrics to evaluate.

    Returns:
      tuple: A tuple containing the loss and losses dictionary.

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
      loss = self.aLossFunc(generatedImageSamples, realImageSamples)
      loss.backward()
      self.optimizer.step()

    losses = {f'{self.modelNames[0]} {k}': v /
              trainLoaderLen for k, v in losses.items()}

    return loss.item(), losses

  def test(self, testLoader: DataLoader,
           metrics: List[str]) -> Dict[str, float]:
    """
    Test the NeuralNetwork model.

    Args:
      testLoader: DataLoader for test data.
      metrics (list): List of metrics to evaluate.

    Returns:
      dict: Dictionary containing the losses.

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

    losses = {f'{self.modelNames[0]} {k}': v /
              testLoaderLen for k, v in losses.items()}
    return losses

  def predict(self, signalSample) -> torch.Tensor:
    """
    Generate an image using the NeuralNetwork model.

    Args:
      signalSample: Input signal sample.

    Returns:
      torch.Tensor: Generated image.

    """
    generatedImage = self.model(signalSample)
    generatedImage = generatedImage.detach().cpu()
    return generatedImage
