import os
from abc import ABC, abstractmethod
from random import choice
from string import ascii_letters
from typing import Literal

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from matplotlib import pyplot as plt
from pandas import DataFrame, concat
from torch import nn
import numpy as np
import torch

from ..IModel import IModel


class IGeneration(IModel):
  def __init__(
      self,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = True,
      oneHotEncode: bool = False,
      downScaleFactor: int = 8,
      **kwargs
  ):
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.downScaleFactor = downScaleFactor

    self.batchSize = kwargs.get('batchSize', 45)
    self.learningRate = kwargs.get('learningRate', 0.0001)
    self.maxEpoch = kwargs.get('maxEpoch', 100)

    self.name = kwargs.get("name", self.__class__.__name__)

    self.device = ""
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")
    print(f"Device: {self.device}")
  
  def run(
    self,
    df_train: DataFrame,
    df_test: DataFrame,
    df_val: DataFrame,
    fixedIndeces: list[int] = None,
    scoring: list = None
  ) -> None:
      
    if scoring is None:
      scoring = [
        "BCE",
        "BCELogits",
        "CE",
        "MSE",
        "L1"
      ]

    # Create a unique ID for the results file and create the results file
    # with the appropriate columns.
    uniqueID = ''.join([choice(ascii_letters) for i in range(10)])
    resultsPath = os.path.join('results', 'generation', f'{self.name} - {uniqueID}.csv')
    df_results = DataFrame({
      'Model Name': [],
      'Learning Rate': [],
      'Down Scale Factor': [],
      'Batch Size': [],
      'Noise': [],
      'Epoch': [],
      **{f'{modelName} {k} Loss': [] for k in scoring for modelName in self.modelNames},
    })

    # Get the values and labels
    trainLoader = self.getLoader(df_train)
    testLoader = self.getLoader(df_test)
    validLoader = self.getLoader(df_val)

    # Get the fixed real images
    if fixedIndeces is None:
      fixedIndeces = [1342, 1239, 119, 1234, 1235, 607, 414, 941]
    fixedRealImages = validLoader.dataset.tensors[0][fixedIndeces]

    imgDir = os.path.join('images', 'generated', self.__class__.__name__, self.name, uniqueID)
    os.makedirs(imgDir, exist_ok=True)
    self.plotImages(imgDir, fixedRealImages, 'original.png', subtitle='Original Images')

    for epoch in range(self.maxEpoch):

      losses = {k: 0 for k in scoring}
      loss = self.train(trainLoader)
      losses = self.test(testLoader, losses)

      print(f'Epoch: {epoch} Loss: {loss}')
      if epoch == 0:
        fixedSignalSamples = validLoader.dataset.tensors[1][fixedIndeces].to(self.device)
      self.predict(fixedSignalSamples, imgDir, epoch)
      

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

  def score(self, scoring: dict, actual, predicted):
    """
    Calculate the scores for different loss functions and update the scoring dictionary.

    Args:
      scoring (dict): A dictionary containing the scores for different loss functions.
      actual: The actual values.
      predicted: The predicted values.

    Returns:
      dict: The updated scoring dictionary.
    """
    newScoring = {k: 0 for k in scoring.keys()}
    BCE = nn.functional.binary_cross_entropy(predicted, actual).item()
    if 'BCE' in scoring.keys(): newScoring['BCE'] += BCE
    BCELogits = nn.functional.binary_cross_entropy_with_logits(predicted, actual).item()
    if 'BCELogits' in scoring.keys(): newScoring['BCELogits'] += BCELogits
    CE = nn.functional.cross_entropy(predicted, actual).item()
    if 'CE' in scoring.keys(): newScoring['CE'] += CE
    MSELoss = nn.functional.mse_loss(predicted, actual).item()
    if 'MSE' in scoring.keys(): newScoring['MSE'] += MSELoss
    L1Loss = nn.functional.l1_loss(predicted, actual).item()
    if 'L1' in scoring.keys(): newScoring['L1'] += L1Loss

    return newScoring
  
  def plotImages(
    self,
    imgDir: str,
    fixedImageSamples,
    fileName: str,
    subtitle: str,
    labels: list[str] = None
  ):
    """
    Plot and save a grid of images.

    Args:
      imgDir (str): The directory where the image will be saved.
      fixedImageSamples: The fixed image samples to be plotted.
      fileName (str): The name of the file to be saved.
      subtitle (str): The title of the plot.

    Returns:
      None
    """
    numRows = len(fixedImageSamples) // 2
    fig, axs = plt.subplots(numRows, 2, figsize=(8, 16))

    imgH = fixedImageSamples.shape[2]
    imgW = fixedImageSamples.shape[3]
    imgMargin = (imgW - imgH) // 2

    for i, ax in enumerate(axs.flatten()):
      image = fixedImageSamples[i][0]
      # crop the image to a square at the center
      image = image[:, imgMargin:imgMargin + imgH]
      ax.imshow(image, cmap='gray', vmin=0, vmax=1)
      ax.axis('off')

    if labels is None:
      labels = [f'({chr(97 + i)})' for i in range(len(fixedImageSamples))]
    for i, ax in enumerate(axs.flatten()):
      ax.set_title(labels[i], fontsize=18)

    # title the plot
    plt.suptitle(subtitle, fontsize=20)
    # save the plot
    savePath = os.path.join(imgDir, fileName)
    plt.savefig(savePath)
    plt.close()

  def getImages(
      self,
      df: DataFrame,
      downScaleFactor: int = 1
  ):
    """
    Get the images from the dataframe

    :param df: dataframe

    :return: images
    """
    # Define the original width and height
    originalWidth = 640
    originalHeight = 480

    # Define the new width and height
    newWidth = originalWidth // downScaleFactor
    newHeight = originalHeight // downScaleFactor

    outputImages = torch.zeros((len(df), 1, newHeight, newWidth))

    for i in range(len(df)):
      row = df.iloc[i]
        # Get the filename
      imageFilename = row['cc_filename']

      imageFilePath = os.path.join(
          'images',
          'processed',
          f'{newHeight}x{newWidth}',
          imageFilename)

      # Read the .png file
      image = plt.imread(imageFilePath)
      # Average the first three channels
      image = np.mean(image[:, :, :3], axis=2)

      # Find all the zero values and replace them with 0.01
      zero_values = image == 0
      image[zero_values] = 0.01

      # Find all the one values nd replace them with 0.99
      one_values = image == 1
      image[one_values] = 0.99

      if image.max() == image.min():
        image = np.ones((newHeight, newWidth)) * 0.99

      # Convert the image to a PyTorch tensor
      image_tensor = torch.from_numpy(image).float()

      # Reshape the tensor to the expected input shape
      image_tensor = image_tensor.view(1, newHeight, newWidth)

      # Add the image to the outputImages array
      outputImages[i] = image_tensor

    return outputImages

  def getLoader(self, df: DataFrame):
    """
    Get the data loader
    """
    values, _ = self.getValuesAndLabels(df)
    images = self.getImages(df, self.downScaleFactor)

    values = torch.from_numpy(values).float()

    dataSet = torch.utils.data.TensorDataset(images, values)

    dataLoader = torch.utils.data.DataLoader(
        dataSet, batch_size=self.batchSize, shuffle=False
    )

    return dataLoader