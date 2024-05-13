import os
from random import choice
from string import ascii_letters
from typing import Literal

from matplotlib import pyplot as plt
from pandas import DataFrame, concat
from torch import nn
import numpy as np
import torch

from ignite.metrics import SSIM
from ignite.engine import Engine

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
    super().__init__()

    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.downScaleFactor = downScaleFactor

    self.perPixelLoss = kwargs.get('perPixelLoss', False)

    self.batchSize = kwargs.get('batchSize', 45)
    self.learningRate = kwargs.get('learningRate', 0.0001)
    self.maxEpoch = kwargs.get('maxEpoch', 100)

    self.loadFile = kwargs.get('loadFile', None)
    self.toSave = kwargs.get('toSave', False)

    self.name = kwargs.get("name", self.__class__.__name__)

    self.device = ""
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")
    print(f"Device: {self.device}")

  def saveModel(self, uniqueID: str) -> None:
    """
    Save the model to the specified path.

    Args:
      path (str): The path to save the model.

    Returns:
      None
    """
    try:
      model_save_dir = os.path.join('models')
      os.makedirs(model_save_dir, exist_ok=True)
      save_path = os.path.join(model_save_dir, f'{self.name} - {uniqueID}.pt')
      torch.save(self.model.state_dict(), save_path)
    except RuntimeError as e:
      print(f'Error saving model: {e}')

  def loadModel(self) -> None:
    """
    Load the model from the specified path.

    Args:
      path (str): The path to load the model from.

    Returns:
      None
    """
    if self.loadFile:
      if self.name not in self.loadFile:
        raise FileExistsError(f'The load file provided {self.loadFile} does not match the model {self.name}')
      load_path = os.path.join(self.loadFile)
      self.model.load_state_dict(torch.load(load_path))
  
  def run(
    self,
    df_train: DataFrame,
    df_test: DataFrame,
    df_val: DataFrame,
    fixedIndeces: list[int] = None,
    metrics: list = None
  ) -> None:

    # Get the values and labels
    trainLoader = self.getLoader(df_train)
    testLoader = self.getLoader(df_test)
    validLoader = self.getLoader(df_val)
      
    if metrics is None:
      metrics = [
        "BCE",
        "BCELogits",
        "MSE",
        "MAE",
        "SSIM"
      ]
    
    if self.perPixelLoss:
      pixelMetrics = ['White MAE', 'Black MAE']
      pixelMetrics = [f'{label} {metric}' for label in self.labelNames for metric in pixelMetrics]
      metrics += pixelMetrics
    
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
      **{f'train {modelName} {k} Loss': [] for k in metrics for modelName in self.modelNames},
      **{f'test {modelName} {k} Loss': [] for k in metrics for modelName in self.modelNames},
    })


    # Get the fixed real images
    if fixedIndeces is None:
      fixedIndeces = [1342, 1239, 119, 1234, 1235, 607, 414, 941]
    
    fixedRealImages = validLoader.dataset.tensors[0][fixedIndeces]

    imgDir = os.path.join('images', 'generated', self.__class__.__name__, self.name, uniqueID)
    os.makedirs(imgDir, exist_ok=True)
    self.plotImages(imgDir, fixedRealImages, 'original.png', subtitle='Original Images')

    self.loadModel()

    min_loss = np.infty

    for epoch in range(self.maxEpoch):

      loss, trainLosses = self.train(trainLoader, metrics)
      testLosses = self.test(testLoader, metrics)

      print(f'Epoch: {epoch} Loss: {loss}')
      self.plotFixedImages(fixedIndeces, validLoader, fixedRealImages, imgDir, epoch)
      

      df_newRow = DataFrame({
        'Model Name': self.__class__.__name__,
        'Learning Rate': self.learningRate,
        'Down Scale Factor': self.downScaleFactor,
        'Batch Size': self.batchSize,
        'Noise': self.noise,
        'Epoch': epoch,
        **{f'train {k} Loss': v for k, v in trainLosses.items()},
        **{f'test {k} Loss': v for k, v in testLosses.items()},
      }, index=[0])
      df_results = concat([df_results, df_newRow], axis=0)
      df_results.to_csv(resultsPath, index=False)
      
      if loss < min_loss and self.toSave:
        min_loss = loss
        self.saveModel(uniqueID)

    self.validate(metrics, validLoader, uniqueID)

  def plotFixedImages(self, fixedIndeces, validLoader, fixedRealImages, imgDir, epoch):
      fixedSignalSamples = validLoader.dataset.tensors[1][fixedIndeces].to(self.device)
      fixedGeneratedImages = self.predict(fixedSignalSamples)
      fixedErrorImages = self.calculatePerPixelLoss(fixedRealImages, fixedGeneratedImages)
      self.plotImages(
        imgDir,
        fixedGeneratedImages,
        f'epoch_{str(epoch).zfill(3)}.png',
        subtitle=f'Epoch {epoch}'
      )
      self.plotImages(
        imgDir,
        fixedErrorImages,
        f'error_epoch_{str(epoch).zfill(3)}.png',
        subtitle=f'Error at Epoch {epoch}',
        palette='viridis',
        colorBar=True,
        colorRange=(-1, 1)
      )

  def validate(self, metrics, validLoader, uniqueID):
    df_val_results = DataFrame()

    for batch in validLoader:
      realImagesSamples, signalSamples, signalLabels, signalIndices = batch
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      generatedImageSamples = self.predict(signalSamples)

      losses = self.singleScore(metrics, realImagesSamples, generatedImageSamples, signalLabels)

    
      labelIndex = torch.argmax(signalLabels, dim=1)
      actualLabels = [self.labelNames[i] for i in labelIndex]
      losses['Label'] = actualLabels
      losses['Index'] = signalIndices

      df_batch = DataFrame(losses)
      df_val_results = concat([df_val_results, df_batch], axis=0)

    # reorder the columns so that the label and index columns are at the beginning
    columns = df_val_results.columns.tolist()
    columns = columns[-2:] + columns[:-2]
    df_val_results = df_val_results[columns]

    # Get all the columns that contain the string 'White MAE'
    whiteMAEColumns = [col for col in df_val_results.columns if 'White MAE' in col]

    # Get all the columns that contain the string 'Black MAE'
    blackMAEColumns = [col for col in df_val_results.columns if 'Black MAE' in col]

    # Sum the columns together
    whiteMAESum = df_val_results[whiteMAEColumns].sum(axis=1)
    blackMAESum = df_val_results[blackMAEColumns].sum(axis=1)

    # Drop the columns that contain the string 'White MAE' and 'Black MAE'
    df_val_results = df_val_results.drop(columns=whiteMAEColumns + blackMAEColumns)

    # Add the summed columns to the dataframe
    df_val_results['White MAE'] = whiteMAESum
    df_val_results['Black MAE'] = blackMAESum

    save_dir = os.path.join('results', 'generation', 'per image')
    os.makedirs(save_dir, exist_ok=True)
    df_val_results.to_csv(os.path.join(save_dir, f'{self.name} - {uniqueID}.csv'), index=False)

  def score(
      self,
      scoring: list,
      actual,
      predicted,
      labels,
      runPerPixelLoss: bool = True,
      dim: tuple = (1, 2, 3)
    ):
    """
    Calculate the scores for different loss functions and update the scoring dictionary.

    Args:
      scoring (dict): A dictionary containing the scores for different loss functions.
      actual: The actual values.
      predicted: The predicted values.

    Returns:
      dict: The updated scoring dictionary.
    """
    newScoring = {k: 0 for k in scoring}

    if self.perPixelLoss and runPerPixelLoss:
      # the actual images are made up of 0.01 and 0.99 values. Replace the 0.01
      # values with 0 and the 0.99 values with 1
      roundActual = actual.clone()
      roundActual[roundActual == 0.01] = 0
      roundActual[roundActual == 0.99] = 1

      predicted = predicted.to(self.device)

      # Calculate the absolute difference between the actual and predicted values
      absDiff = torch.abs(roundActual - predicted)

      # White MAE loss is the mean of the absolute difference between the actual
      # and predicted values for the pixels that are 0.99 (1 in the rounded
      # roundActual tensor)

      numWhitePixels = torch.sum(roundActual, dim=dim)
      whiteAE = absDiff * roundActual
      sumWhiteAE = torch.sum(whiteAE, dim=dim)
      whiteMAE = sumWhiteAE / numWhitePixels
      whiteMAE = whiteMAE.detach().cpu().numpy()

      # Black MAE loss is the mean of the absolute difference between the actual
      # and predicted values for the pixels that are 0.01 (0 in the rounded
      # roundActual tensor)
      numBlackPixels = torch.sum(1 - roundActual, dim=dim)
      blackAE = absDiff * (1 - roundActual)
      sumBlackAE = torch.sum(blackAE, dim=dim)
      blackMAE = sumBlackAE / numBlackPixels
      blackMAE = blackMAE.detach().cpu().numpy()

      # the labels are one-hot encoded. Get the index of the label that is 1
      labelIndex = torch.argmax(labels, dim=dim[0])
      
      # self.labelNames in a list of the label names. labelIndex is a tensor
      # containing the indexes of the labels in self.labelNames that correspond
      # to the actual image. Get the label names for the actual images
      if labelIndex.dim() == 0:
        labelIndex = labelIndex.item()
        actualLabels = self.labelNames[labelIndex]
      else:
        actualLabels = [self.labelNames[i] for i in labelIndex]
      actualLabels = np.array(actualLabels)

      # Create a dataframe with one column for the actual labels, one column for
      # the white MAE loss, and one column for the black MAE loss.
      if whiteMAE.shape == () and blackMAE.shape == () and actualLabels.shape == ():
        df = DataFrame({
          'Actual Labels': actualLabels,
          'White MAE': whiteMAE,
          'Black MAE': blackMAE
        }, index=[0])
      else:
        df = DataFrame({
          'Actual Labels': actualLabels,
          'White MAE': whiteMAE,
          'Black MAE': blackMAE
        })

      # Group the dataframe by the actual labels and calculate the mean of the
      # white MAE and black MAE losses for each label
      df = df.groupby('Actual Labels').mean()

      # replace NaN values with 0
      df = df.fillna(0)

      for row in df.iterrows():
        label = row[0]
        whiteMAE = row[1]['White MAE']
        blackMAE = row[1]['Black MAE']

        newScoring[f'{label} White MAE'] += whiteMAE
        newScoring[f'{label} Black MAE'] += blackMAE

    
    BCE = nn.functional.binary_cross_entropy(predicted, actual).item()
    if 'BCE' in scoring: newScoring['BCE'] += BCE

    BCELogits = nn.functional.binary_cross_entropy_with_logits(predicted, actual).item()
    if 'BCELogits' in scoring: newScoring['BCELogits'] += BCELogits

    MSELoss = nn.functional.mse_loss(predicted, actual).item()
    if 'MSE' in scoring: newScoring['MSE'] += MSELoss

    MAELoss = nn.functional.l1_loss(predicted, actual).item()
    if 'MAE' in scoring: newScoring['MAE'] += MAELoss

    if runPerPixelLoss: 
      # SSIM loss
      def eval_step(engine, batch):
        return batch

      default_evaluator = Engine(eval_step)

      ssim = SSIM(data_range=1.0)
      ssim.attach(default_evaluator, 'SSIM')

      if len(predicted.shape) == 3:
        predicted = predicted.unsqueeze(0)

      if len(actual.shape) == 3:
        actual = actual.unsqueeze(0)


      state = default_evaluator.run([[predicted, actual]])
      SSIMLoss = state.metrics['SSIM']
      if 'SSIM' in scoring: newScoring['SSIM'] += SSIMLoss

    return newScoring
  
  def singleScore(self, scoring: list, actual, predicted, labels, runPerPixelLoss: bool = True):
    """
    Calculate the scores for different loss functions and update the scoring dictionary.

    Args:
      scoring (dict): A dictionary containing the scores for different loss functions.
      actual: The actual values.
      predicted: The predicted values.

    Returns:
      dict: The updated scoring dictionary.
    """
    newScoring = {k: [] for k in scoring}

    for i in range(len(actual)):
      singleScoring = self.score(
        scoring,
        actual[i],
        predicted[i],
        labels[i],
        runPerPixelLoss,
        dim=(0, 1, 2)
      )
      for k, v in singleScoring.items():
        newScoring[k].append(v)

    return newScoring
  
  def calculatePerPixelLoss(self, actual, predicted):
    """
    Calculate the per pixel loss.

    Args:
      actual: The actual values.
      predicted: The predicted values.

    Returns:
      dict: The per pixel loss.
    """
    # the actual images are made up of 0.01 and 0.99 values. Replace the 0.01
    # values with 0 and the 0.99 values with 1
    roundActual = actual.clone()
    roundActual[roundActual == 0.01] = 1
    roundActual[roundActual == 0.99] = 0
    roundActual = roundActual.to(self.device)

    roundPredicted = predicted.clone()
    roundPredicted = 1 - roundPredicted
    roundPredicted = roundPredicted.to(self.device)

    # Calculate the absolute difference between the actual and predicted values
    diff = roundActual - roundPredicted

    return diff
  
  def plotImages(
    self,
    imgDir: str,
    fixedImageSamples,
    fileName: str,
    subtitle: str,
    labels: list[str] = None,
    palette: str = 'gray',
    colorBar: bool = False,
    colorRange: tuple = (0, 1),
    numCols: int = 2
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
    # check if fixedImageSamples is a tensor
    if not torch.is_tensor(fixedImageSamples):
      # convert numpy array to tensor
      fixedImageSamples = torch.from_numpy(fixedImageSamples)

    if fixedImageSamples.ndim == 3:
      fixedImageSamples = fixedImageSamples.unsqueeze(1)

    numRows = len(fixedImageSamples) // numCols
    fig, axs = plt.subplots(numRows, numCols, figsize=(4 * numCols, 3.75 * numRows))

    imgH = fixedImageSamples.shape[2]
    imgW = fixedImageSamples.shape[3]
    imgMargin = (imgW - imgH) // 2

    fixedImageSamples = fixedImageSamples.clone()
    fixedImageSamples = fixedImageSamples[:, :, :, imgMargin:imgMargin + imgH]


    if numRows == 1 and numCols == 1:
      image = fixedImageSamples[0][0]
      plt.imshow(image, cmap=palette, vmin=colorRange[0], vmax=colorRange[1])
      plt.axis('off')

      if colorBar:
        cbar = plt.colorbar(
          plt.imshow(fixedImageSamples[0][0], cmap=palette, vmin=colorRange[0], vmax=colorRange[1]),
          orientation='horizontal',
          ticks=[colorRange[0], 0, colorRange[1]],
          label='Error',
        )
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.set_xlabel('Error', fontsize=18)
    else:
      for i, ax in enumerate(axs.flatten()):
        image = fixedImageSamples[i][0]
        image = image.cpu()
        ax.imshow(image, cmap=palette, vmin=colorRange[0], vmax=colorRange[1])
        ax.axis('off')

      if labels is None:
        labels = [f'({chr(97 + i)})' for i in range(len(fixedImageSamples))]
      for i, ax in enumerate(axs.flatten()):
        ax.set_title(labels[i], fontsize=18)

      if colorBar:
        # Add a color bar to the bottom of the plot if colorBar is True
        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])
        image = fixedImageSamples[0][0]
        image = image.cpu()
        cbar = fig.colorbar(
          axs[0, 0].imshow(image, cmap=palette, vmin=colorRange[0], vmax=colorRange[1]), 
          cax=cbar_ax,
          orientation='horizontal',
          ticks=[colorRange[0], 0, colorRange[1]],
          label='Error',
        )
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.set_xlabel('Error', fontsize=18)

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
    values, labels = self.getValuesAndLabels(df)
    indicies = self.getIndices(df)
    images = self.getImages(df, self.downScaleFactor)

    values = torch.from_numpy(values).float()
    indicies = torch.from_numpy(indicies).int()

    if not self.oneHotEncode and self.perPixelLoss:
      raise ValueError("oneHotEncode must be True")
    else:
      labels = torch.from_numpy(labels).int()

    dataSet = torch.utils.data.TensorDataset(images, values, labels, indicies)

    dataLoader = torch.utils.data.DataLoader(
        dataSet, batch_size=self.batchSize, shuffle=False
    )

    return dataLoader