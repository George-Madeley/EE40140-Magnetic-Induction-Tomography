import os
from random import choice
from string import ascii_letters
from typing import Literal

import torch
from torch import nn
from torch.optim import Adam
from pandas import DataFrame, concat

from .ArtificialNeuralNetwork import ANN
from ..IModel import IModel

class NeuralNetwork(IModel):
  def __init__(
    self,
    labelName: Literal['shape', 'sample'] = 'shape',
    noise: bool = False,
    oneHotEncode: bool = False,
    downScaleFactor: int = 32,
    structure: list[int] = None,
    **kwargs,
  ):
    
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.downScaleFactor = downScaleFactor

    self.batchSize = kwargs.get("batchSize")
    self.learningRate = kwargs.get("learningRate")
    self.maxEpoch = kwargs.get("maxEpochs")

    self.name = kwargs.get("name", self.__class__.__name__)

    self.device = ""
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")

    self.network = ANN(
      inputSize= 240 if noise else 120,
      width=640 // downScaleFactor,
      height=480 // downScaleFactor,
      structure=structure,
    ).to(self.device)

    self.optimizer = Adam(
      self.network.parameters(),
      lr=self.learningRate,
    )

  def train(
    self,
    trainLoader,
  ):
    
    for realImageSamples, signalSamples in trainLoader:
      # Create and label the real samples, the generated samples, and the
      # latent space samples. Send them to the chosen device i.e., CPU or GPU.
      realImageSamples = realImageSamples.to(device=self.device)
      signalSamples = signalSamples.to(device=self.device)

      # Zero the gradients of the optimizer.
      self.optimizer.zero_grad()

      # Generate the fake samples
      generatedImageSamples = self.network(signalSamples)

      # Calculate the loss
      loss = nn.BCELoss()(generatedImageSamples, realImageSamples)
      loss.backward()
      self.optimizer.step()

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

      generatedImageSamples = self.network(signalSamples)

      losses = super().score(losses, realImagesSamples, generatedImageSamples)
    
    losses = {k: v / testLoaderLen for k, v in losses.items()}
    return losses
  
  def predict(self, df: DataFrame) -> None:
    raise NotImplementedError
  
  def run(
    self,
    df_train: DataFrame,
    df_test: DataFrame,
    fixedIndeces: list[int] = None,
  ) -> None:
      
    losses = {
      "BCE": 0,
      "BCELogits": 0,
      "CE": 0,
      "MSE": 0,
      "L1": 0,
    }

    uniqueID = ''.join([choice(ascii_letters) for i in range(10)])
    resultsPath = os.path.join('results', 'generation', f'{self.name} - {uniqueID}.csv')
    df_results = DataFrame({
      'Model Name': [],
      'Learning Rate': [],
      'Down Scale Factor': [],
      'Batch Size': [],
      'Noise': [],
      'Epoch': [],
      **{f'{k} Loss': [] for k in losses.keys()},
    })

    # Get the values and labels
    trainLoader = super().getLoader(df_train)
    testLoader = super().getLoader(df_test)

    if fixedIndeces is None:
      fixedIndeces = [1342, 1239, 119, 1234, 1235, 607, 414, 941]
    fixedRealImages = testLoader.dataset.tensors[0][fixedIndeces]

    imgDir = os.path.join('images', 'generated', self.__class__.__name__, self.name, uniqueID)
    os.makedirs(imgDir, exist_ok=True)
    super().plotImages(imgDir, fixedRealImages, 'original.png', subtitle='Original Images')

    for epoch in range(self.maxEpoch):

      losses = {k: 0 for k in losses.keys()}

      loss = self.train(trainLoader)
      losses = self.test(
        testLoader,
        losses
      )

      print(f'Epoch: {epoch}\tLoss NN: {loss}')
      if epoch == 0:
        fixedSignalSamples = testLoader.dataset.tensors[1][fixedIndeces].to(self.device)
      fixedGeneratedImages = self.network(fixedSignalSamples)
      fixedGeneratedImages = fixedGeneratedImages.detach().cpu()
      self.plotImages(imgDir, fixedGeneratedImages, f'epoch_{str(epoch).zfill(3)}.png', subtitle=f'Epoch {epoch}')

      df_newRow = DataFrame({
        'Model Name': self.__class__.__name__,
        'Learning Rate': self.learningRate,
        'Down Scale Factor': self.downScaleFactor,
        'Batch Size': self.batchSize,
        'Noise': self.noise,
        'Epoch': epoch,
        **{f'{k} Loss': v for k, v in losses.items()},
      }, index=[0])
      df_results = concat([df_results, df_newRow], axis=0)
      df_results.to_csv(resultsPath, index=False)
  
  @staticmethod
  def getDefaultParams():
    """
    Get the default parameters

    :return: default parameters
    """
    defaultParams = {
      "learningRate": 0.0001,
      "numEpochs": 100,
      "batchSize": 45,
      "lossFunc": nn.BCELoss(),
      "downScaleFactor": 32,
      "noise": False,
    }
    return defaultParams