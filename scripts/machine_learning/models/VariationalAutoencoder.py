import os
from random import choice
from string import ascii_letters
from typing import Literal

from pandas import DataFrame, concat, read_csv
import torch
from .IModel import IModel
from .Encoder import Encoder
from .Decoder import Decoder

class VariationalAutoencoder(IModel):
  def __init__(
    self,
    labelName: Literal['shape', 'sample'] = 'shape',
    noise: bool = False,
    oneHotEncode: bool = False,
    downScaleFactor: int = 32,
    **kwargs
  ):
    self.validParams = {
      'downScaleFactor': [32, 20, 16, 10, 8, 5, 4, 2, 1],
      'learningRate': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
      'maxEpoch': list(range(1, 1001)),
    }
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.downScaleFactor = downScaleFactor

    defaultParams = VariationalAutoencoder.getDefaultParams()

    self.batchSize = kwargs.get('batchSize', defaultParams['batchSize'])
    self.learningRate = kwargs.get('learningRate', defaultParams['learningRate'])
    self.maxEpoch = kwargs.get('maxEpoch', defaultParams['maxEpoch'])

    self.device = ""
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")

    latent_dim = 256
    self.encoder = Encoder(
      240 if noise else 120,
      latent_dim
    ).to(self.device)
    self.decoder = Decoder(
      latent_dim,
      640 // downScaleFactor,
      480 // downScaleFactor 
    )

    self.optimizerEncoder = torch.optim.Adam(
      self.encoder.parameters(),
      lr=self.learningRate
    )
    self.optimizerDecoder = torch.optim.Adam(
      self.decoder.parameters(),
      lr=self.learningRate
    )

  def train(
    self,
    trainLoader
  ) -> None:
    for realImagesSamples, signalSamples in trainLoader:
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      self.optimizerEncoder.zero_grad()
      self.optimizerDecoder.zero_grad()

      latentSamples = self.encoder(signalSamples)
      generatedImageSamples = self.decoder(latentSamples)

      loss = ((realImagesSamples - generatedImageSamples) ** 2).sum() + self.encoder.kl.sum()
      loss.backward()
      self.optimizerEncoder.step()
      self.optimizerDecoder.step()

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

      latentSamples = self.encoder(signalSamples)
      generatedImageSamples = self.decoder(latentSamples)

      losses = super().score(losses, realImagesSamples, generatedImageSamples)
    
    losses = {k: v / testLoaderLen for k, v in losses.items()}
    return losses
  
  def predict(self, df: DataFrame) -> None:
    raise NotImplementedError
  
  def varyParams(
    self,
    df_train: DataFrame,
    df_test: DataFrame,
    params: dict,
  ) -> None:
    
    validParams = {}
    for param in params.keys():
      if super().isParamValid(param):
        validParams[param] = params[param]

    if validParams == {}:
      print(
        f'No valid parameters to vary for for {self.__class__.__name__} model.')
      
    losses = {
      "BCE": 0,
      "BCELogits": 0,
      "CE": 0,
      "MSE": 0,
      "L1": 0,
    }

    if os.path.exists(os.path.join('results', f'{self.__class__.__name__}.csv')):
      df_results = read_csv(
        os.path.join('results', f'{self.__class__.__name__}.csv')
      )
    else:
      df_results = DataFrame({
        'Model Name': [],
        'Learning Rate': [],
        'Down Scale Factor': [],
        'Max Epoch': [],
        'Batch Size': [],
        'Noise': [],
        'Epoch': [],
        **{f'{k} Loss': [] for k in losses.keys()},
        'Img ID': [],
      })

    for paramName in validParams.keys():
      for param in validParams[paramName]:
        # Get the values and labels
        trainLoader = super().getLoader(df_train)
        testLoader = super().getLoader(df_test)

        fixedIndeces = [1342, 941, 119, 823, 607, 414]
        fixedRealImages = testLoader.dataset.tensors[0][fixedIndeces]

        uniqueID = ''.join([choice(ascii_letters) for i in range(10)])
        imgDir = os.path.join('images', 'epochs', f'{self.__class__.__name__}', uniqueID)
        os.makedirs(imgDir, exist_ok=True)
        super().plotImages(imgDir, fixedRealImages, 'original.png', subtitle='Original Images')
        resultsPath = os.path.join('results', f'{self.__class__.__name__}.csv')

        defaultParams = self.getDefaultParams()
        self.__init__(**defaultParams)
        setattr(self, paramName, param)

        for epoch in range(self.maxEpoch):

          losses = {k: 0 for k in losses.keys()}

          loss = self.train(trainLoader)
          losses = self.test(
            testLoader,
            losses
          )

          print(f'Epoch: {epoch}\tLoss E.: {loss}\tLoss D.: {loss}')
          if epoch == 0:
            fixedSignalSamples = testLoader.dataset.tensors[1][fixedIndeces].to(self.device)
          fixedLatentSamples = self.encoder(fixedSignalSamples)
          fixedGeneratedImages = self.decoder(fixedLatentSamples)
          fixedGeneratedImages = fixedGeneratedImages.detach().cpu()
          self.plotImages(imgDir, fixedGeneratedImages, f'epoch_{str(epoch).zfill(3)}.png', subtitle=f'Epoch {epoch}')

          df_newRow = DataFrame({
            'Model Name': self.__class__.__name__,
            'Learning Rate': self.learningRate,
            'Down Scale Factor': self.downScaleFactor,
            'Max Epoch': self.maxEpoch,
            'Batch Size': self.batchSize,
            'Noise': self.noise,
            'Epoch': epoch,
            **{f'{k} Loss': v for k, v in losses.items()},
            'Img ID': uniqueID,
          }, index=[0])
          df_results = concat([df_results, df_newRow], axis=0)
          df_results.to_csv(resultsPath, index=False)

  @staticmethod
  def getDefaultParams():
    defaultParams = {
      "learningRate": 0.0001,
      "maxEpoch": 100,
      "batchSize": 45,
      "downScaleFactor": 32,
      "noise": False,
    }
    return defaultParams
