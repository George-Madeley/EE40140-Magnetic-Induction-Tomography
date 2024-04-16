import os
from random import choice
from string import ascii_letters
from typing import Literal

import torch
from torch import nn
from torch.optim import Adam
from pandas import DataFrame, concat, read_csv

from .Discriminator import Discriminator
from .Generator import Generator
from ..IModel import IModel


class GenerativeAdversarialNetwork(IModel):
  def __init__(
      self,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = False,
      oneHotEncode: bool = False,
      downScaleFactor: int = 32,
      **kwargs
  ):
    """
    Initialize a GenerativeAdversarialNetwork object.

    Args:
        labelName (Literal['shape', 'sample'], optional): The name of the label to predict. Defaults to 'shape'.
        noise (bool, optional): Whether to add noise to the data. Defaults to False.
    """

    self.validParams = {
      'learningRate': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
      'maxEpoch': list(range(1, 1001)),
      'dLossFunc': [nn.BCELoss(), nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.MSELoss(), nn.L1Loss()],
      'gLossFunc': [nn.BCELoss(), nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.MSELoss(), nn.L1Loss()],
    }
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.downScaleFactor = downScaleFactor

    defaultParams = GenerativeAdversarialNetwork.getDefaultParams()

    self.batchSize = kwargs.get("batchSize", defaultParams["batchSize"])
    self.learningRate = kwargs.get("learningRate", defaultParams["learningRate"])
    self.maxEpoch = kwargs.get("numEpochs", defaultParams["numEpochs"])
    self.DLossFunc = kwargs.get("dLossFunc",defaultParams["dLossFunc"])
    self.GLossFunc = kwargs.get("gLossFunc",defaultParams["gLossFunc"])

    self.device = ""
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")

    self.discriminator = Discriminator(
      640 // downScaleFactor,
      480 // downScaleFactor
    ).to(self.device)
    self.generator = Generator(
      240 if noise else 120,
      640 // downScaleFactor,
      480 // downScaleFactor
    ).to(self.device)

    self.optimizerDiscriminator = Adam(
      self.discriminator.parameters(),
      lr=self.learningRate
    )
    self.optimizerGenerator = Adam(
      self.generator.parameters(),
      lr=self.learningRate
    )

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
    
    for realSamples, latentSpaceSamples in trainLoader:
      # Create and label the real samples, the generated samples, and the
      # latent space samples. Send them to the chosen device i.e., CPU or GPU.
      realSamples = realSamples.to(device=self.device)
      realSampleLabels = torch.ones((self.batchSize, 1)).to(device=self.device)
      latentSpaceSamples = latentSpaceSamples.to(device=self.device)
      generatedSamples = self.generator(latentSpaceSamples)
      generatedSampleLabels = torch.zeros((self.batchSize, 1)).to(device=self.device)
      allSamples = torch.cat((realSamples, generatedSamples))
      allSampleLabels = torch.cat((realSampleLabels, generatedSampleLabels))

      # Training the discriminator
      self.discriminator.zero_grad()
      outputDiscriminator = self.discriminator(allSamples)
      lossDiscriminator = self.DLossFunc(
          outputDiscriminator, allSampleLabels
      )
      lossDiscriminator.backward()
      self.optimizerDiscriminator.step()

      # Training the generator
      self.generator.zero_grad()
      generatedSamples = self.generator(latentSpaceSamples)
      lossGenerator = self.GLossFunc(
          generatedSamples, realSamples
      )
      lossGenerator.backward()
      self.optimizerGenerator.step()

    return lossDiscriminator, lossGenerator

  def test(
      self,
      testLoader,
      discriminatorLosses: dict,
      generatorLosses: dict,
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
      discriminatorLosses = super().score(discriminatorLosses, allSampleLabels, outputDiscriminator)

      # Test the generator
      self.generator.zero_grad()
      generatedSamples = self.generator(latentSpaceSamples)
      generatorLosses = super().score(generatorLosses, realSamples, generatedSamples)

    discriminatorLosses = {k: v / testLoaderLen for k, v in discriminatorLosses.items()}
    generatorLosses = {k: v / testLoaderLen for k, v in generatorLosses.items()}

    return discriminatorLosses, generatorLosses

  def predict(self, df: DataFrame) -> None:
    """
    Predict the labels of the test data

    :param predict_df: prediction dataframe

    :return: predictions
    """
    raise NotImplementedError
    pass

  def varyParams(
      self,
      df_train: DataFrame,
      df_test: DataFrame,
      params: dict,
    ) -> None:
    """
    Vary the parameters of the model

    :param params: parameters
    """
    validParams = {}
    for param in params.keys():
      if super().isParamValid(param):
        validParams[param] = params[param]

    if validParams == {}:
      print(
        f'No valid parameters to vary for for {self.__class__.__name__} model.')
      
    discriminatorLosses = {
      "BCE": 0,
      "BCELogits": 0,
      "CE": 0,
      "MSE": 0,
      "L1": 0,
    }

    generatorLosses = {
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
        **{f'D {k} Loss': [] for k in discriminatorLosses.keys()},
        **{f'G {k} Loss': [] for k in generatorLosses.keys()},
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

          discriminatorLosses = {k: 0 for k in discriminatorLosses.keys()}
          generatorLosses = {k: 0 for k in generatorLosses.keys()}

          discriminatorLoss, generatorLoss = self.train(trainLoader)
          discriminatorLosses, generatorLosses = self.test(
            testLoader,
            discriminatorLosses,
            generatorLosses,
          )

          print(f'Epoch: {epoch}\tLoss D.: {discriminatorLoss}\tLoss G.: {generatorLoss}')
          if epoch == 0:
            fixedLatentSamples = testLoader.dataset.tensors[1][fixedIndeces].to(self.device)
          fixedGeneratedImages = self.generator(fixedLatentSamples)
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
            **{f'D {k} Loss': v for k, v in discriminatorLosses.items()},
            **{f'G {k} Loss': v for k, v in generatorLosses.items()},
            'Img ID': uniqueID,
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
      "dLossFunc": nn.BCELoss(),
      "gLossFunc": nn.MSELoss(),
      "downScaleFactor": 32,
      "noise": False,
    }
    return defaultParams
