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


class IModel(ABC):
  @abstractmethod
  def train(self, df: DataFrame) -> None:
    """
    Train the model

    :param train_df: training dataframe
    """
    pass

  @abstractmethod
  def test(self, df: DataFrame) -> None:
    """
    Test the model

    :param test_df: test dataframe

    :return: metrics
    """
    pass

  @abstractmethod
  def predict(self, df: DataFrame) -> None:
    """
    Predict the labels of the test data

    :param predict_df: prediction dataframe

    :return: predictions
    """
    pass

  @abstractmethod
  def getDefaultParams(self):
    """
    Get the default parameters

    :return: default parameters
    """
    pass

  def varyParams(
      self,
      df: DataFrame,
      params: dict,
      searchCV: type[GridSearchCV | RandomizedSearchCV],
      scoring: dict = None,
      n_jobs: int = -1,
      verbose: int = 0,
      material: Literal['iron', 'copper'] = 'iron',
      numSamples: Literal[120, 240] = 120,
    ) -> None:
    """
    Vary the parameters of the model

    :param params: parameters
    """
    validParams = {}
    for param in params.keys():
      if self.isParamValid(param):
        validParams[param] = params[param]

    if validParams == {}:
      print(
        f'No valid parameters to vary for for {self.__class__.__name__} model.')

    if scoring is None:
      scoring = {
        'Accuracy': 'accuracy',
        'F1': 'f1_micro',
        'Precision': 'precision_micro',
        'Recall': 'recall_micro',
      }

    values, labels = self.getValuesAndLabels(df)

    clf = searchCV(
      self.model,
      validParams,
      scoring=scoring,
      n_jobs=n_jobs,
      verbose=verbose,
      refit=False
    )
    clf.fit(values, labels)

    df_results = concat([
      DataFrame(clf.cv_results_['params']),
      DataFrame({metric: clf.cv_results_[f'mean_test_{metric}'] for metric in scoring.keys()}),
      DataFrame(clf.cv_results_['mean_fit_time'], columns=['mean_fit_time']),
      DataFrame(clf.cv_results_['std_fit_time'], columns=['std_fit_time']),
      DataFrame(clf.cv_results_['mean_score_time'], columns=['mean_score_time']),
      DataFrame(clf.cv_results_['std_score_time'], columns=['std_score_time'])
    ], axis=1)

    # Add the column 'model' to the dataframe and set it to the model name
    df_results.insert(0, 'model', self.__class__.__name__)
    df_results.insert(1, 'material', material)
    df_results.insert(2, 'numSamples', numSamples)

    # Generate a random string
    randomString = ''.join(
      [choice(ascii_letters) for i in range(10)]
    )

    saveFilePath = os.path.join(
        'results', f'{self.__class__.__name__}_results_{randomString}.csv')
    # Save the results to a CSV file
    df_results.to_csv(
      saveFilePath,
      index=False
    )


  def isParamValid(self, name: str, value=None):
    """
    Check if the parameter is valid

    :param name: name of the parameter
    :param value: value of the parameter

    :return: boolean
    """
    if name in self.validParams.keys():
      return True
    return False
  
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
  
    BCE = nn.functional.binary_cross_entropy(predicted, actual).item()
    if 'BCE' in scoring.keys(): scoring['BCE'] += BCE
    BCELogits = nn.functional.binary_cross_entropy_with_logits(predicted, actual).item()
    if 'BCELogits' in scoring.keys(): scoring['BCELogits'] += BCELogits
    CE = nn.functional.cross_entropy(predicted, actual).item()
    if 'CE' in scoring.keys(): scoring['CE'] += CE
    MSELoss = nn.functional.mse_loss(predicted, actual).item()
    if 'MSE' in scoring.keys(): scoring['MSE'] += MSELoss
    L1Loss = nn.functional.l1_loss(predicted, actual).item()
    if 'L1' in scoring.keys(): scoring['L1'] += L1Loss

    return scoring
  
  def plotImages(
    self,
    imgDir: str,
    fixedImageSamples,
    fileName: str,
    subtitle: str
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
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    for i, ax in enumerate(axs.flatten()):
      ax.imshow(fixedImageSamples[i][0], cmap='gray', vmin=0, vmax=1)
      ax.axis('off')
    plt.tight_layout()

    # title the plot
    plt.suptitle(subtitle)
    # save the plot
    savePath = os.path.join(imgDir, fileName)
    plt.savefig(savePath)
    plt.close()

  def getValuesAndLabels(
    self,
    df: DataFrame,
  ):
    """
    Get the values and labels from the dataframe

    :param df: dataframe

    :return: values, labels
    """

    valueColumnNames = df.filter(regex='cc_\d+').columns
    if self.noise:
      valueColumnNames = valueColumnNames.append(df.filter(regex='bb_\d+').columns)

    values = df[valueColumnNames].values

    if self.oneHotEncode:
      labelColumnNames = df.filter(regex=f'^{self.labelName}_').columns
      labels = df[labelColumnNames].values
    else:
      labels = df[self.labelName].values

    return values, labels

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
        dataSet, batch_size=self.batchSize, shuffle=True
    )

    return dataLoader