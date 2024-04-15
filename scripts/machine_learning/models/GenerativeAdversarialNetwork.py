import csv
import os
from typing import Literal
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

from pandas import DataFrame
from .Discriminator import Discriminator
from .Generator import Generator

from .IModel import IModel


class GenerativeAdversarialNetwork(IModel):
  def __init__(
      self,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = False,
      oneHotEncode: bool = False,
      downScaleFactor: int = 1,
      **kwargs
  ):
    """
    Initialize a GenerativeAdversarialNetwork object.

    Args:
        labelName (Literal['shape', 'sample'], optional): The name of the label to predict. Defaults to 'shape'.
        noise (bool, optional): Whether to add noise to the data. Defaults to False.
    """

    originalWidth = 640
    originalHeight = 480

    defaultParams = GenerativeAdversarialNetwork.getDefaultParams()

    self.learningRate = kwargs.get(
        "learningRate", defaultParams["learningRate"])
    self.numEpochs = kwargs.get("numEpochs", defaultParams["numEpochs"])
    self.batchSize = kwargs.get("batchSize", defaultParams["batchSize"])
    self.lossFunctionDiscriminator = kwargs.get(
        "lossFunctionDiscriminator",
        defaultParams["lossFunctionDiscriminator"])
    self.lossFunctionGenerator = kwargs.get(
        "lossFunctionGenerator",
        defaultParams["lossFunctionGenerator"])

    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.downScaleFactor = downScaleFactor

    self.device = ""
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")

    self.discriminator = Discriminator(
      originalWidth // downScaleFactor,
      originalHeight // downScaleFactor
    ).to(self.device)
    self.generator = Generator(
      240 if noise else 120,
      originalWidth // downScaleFactor,
      originalHeight // downScaleFactor
    ).to(self.device)

  def train(self, df: DataFrame, imgDir: str, dfPath: str) -> None:
    """
    Train the generative adversarial network model using the provided DataFrame.

    Args:
        df (DataFrame): The input DataFrame containing the training data.

    Returns:
        None
    """
    print(
      f"Training the generative adversarial network model on {self.device}...")

    values, _ = self.getValuesAndLabels(df)
    images = self.getImages(df, self.downScaleFactor)

    values = torch.from_numpy(values).float()

    trainSet = torch.utils.data.TensorDataset(images, values)

    trainLoader = torch.utils.data.DataLoader(
        trainSet, batch_size=self.batchSize, shuffle=True
    )

    numSamples = len(df)
    numBatches = numSamples // self.batchSize
    randomSampleIndeces = [1342, 941, 119, 823, 607, 414]

    fixedImageSamples = trainLoader.dataset.tensors[0][randomSampleIndeces]
    # Plot and save the generated images
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    for i, ax in enumerate(axs.flatten()):
      ax.imshow(fixedImageSamples[i][0], cmap='gray', vmin=0, vmax=1)
      ax.axis('off')
    plt.tight_layout()

    # title the plot
    plt.suptitle(f"Orignal Images")
    # save the plot
    savePath = os.path.join(imgDir, f'original.png')
    plt.savefig(savePath)

    # define the discriminator and generator optimizers
    optimizerDiscriminator = Adam(
      self.discriminator.parameters(),
      lr=self.learningRate
    )
    optimizerGenerator = Adam(
      self.generator.parameters(),
      lr=self.learningRate
    )

    print("Training the generative adversarial network model...")
    for epoch in range(self.numEpochs):
      for n, (realSamples, latentSpaceSamples) in enumerate(trainLoader):
        # Get the real samples and send to device
        realSamples = realSamples.to(
            device=self.device
        )

        # Create the labels which are later used as input for the BCE loss
        # function for the discriminator and send to device
        realSampleLabels = torch.ones((self.batchSize, 1)).to(
            device=self.device
        )

        # Send the latent samples to the device
        latentSpaceSamples = latentSpaceSamples.to(
            device=self.device
        )

        # Generate the fake samples from the latent samples
        generatedSamples = self.generator(latentSpaceSamples)

        # Create the labels for the fake samples
        generatedSampleLabels = torch.zeros((self.batchSize, 1)).to(
            device=self.device
        )

        # Combine the real and fake samples
        allSamples = torch.cat((realSamples, generatedSamples))
        allSampleLabels = torch.cat(
            (realSampleLabels, generatedSampleLabels)
        )

        # Training the discriminator
        self.discriminator.zero_grad()
        outputDiscriminator = self.discriminator(allSamples)
        lossDiscriminator = self.lossFunctionDiscriminator(
            outputDiscriminator, allSampleLabels
        )
        lossDiscriminator.backward()
        optimizerDiscriminator.step()

        # Training the generator
        self.generator.zero_grad()
        generatedSamples = self.generator(latentSpaceSamples)
        lossGenerator = self.lossFunctionGenerator(
            generatedSamples, realSamples
        )
        lossGenerator.backward()
        optimizerGenerator.step()

      print(f"Epoch: {epoch} Loss D.: {lossDiscriminator}")
      print(f"Epoch: {epoch} Loss G.: {lossGenerator}")
      print("--------------------------------------------------")

      # Generate images using random latent samples
      if epoch == 0:
        fixedLatentSamples = trainLoader.dataset.tensors[1][randomSampleIndeces].to(
          self.device)
      generatedImages = self.generator(fixedLatentSamples)
      generatedImages = generatedImages.detach().cpu()

      # Plot and save the generated images
      fig, axs = plt.subplots(2, 3, figsize=(8, 6))
      for i, ax in enumerate(axs.flatten()):
        ax.imshow(generatedImages[i][0], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
      plt.tight_layout()

      # title the plot
      plt.suptitle(f"Epoch {epoch}")
      # save the plot
      savePath = os.path.join(imgDir, f'epoch_{str(epoch).zfill(3)}.png')
      plt.savefig(savePath)
      plt.close()

      uniqueID = imgDir.split('/')[-1]

      with open(dfPath, 'a', newline='') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow([
          self.__class__.__name__,
          self.learningRate,
          self.downScaleFactor,
          self.numEpochs,
          self.batchSize,
          epoch,
          lossDiscriminator.item(),
          lossGenerator.item(),
          uniqueID,
          self.noise
        ])

  def test(self, df: DataFrame) -> float:
    """
    Test the generative adversarial network model on the given DataFrame and return the accuracy score.

    Parameters:
    - df (DataFrame): The DataFrame containing the test data.

    Returns:
    - float: The accuracy score of the generative adversarial network model on the test data.
    """
    values, _ = self.getValuesAndLabels(df)
    images = self.getImages(df, self.downScaleFactor)

    testSet = torch.utils.data.TensorDataset(images, values)

    testLoader = torch.utils.data.DataLoader(
        testSet, batch_size=self.batchSize, shuffle=True
    )
    totalLossGenerator = 0
    for realSamples, latentSpaceSamples in testLoader:
      # Get the real samples and send to device
      realSamples = realSamples.to(
          device=self.device
      )

      # Send the latent samples to the device
      latentSpaceSamples = latentSpaceSamples.to(
          device=self.device
      )

      # Test the generator
      self.generator.zero_grad()
      generatedSamples = self.generator(latentSpaceSamples)
      lossGenerator = self.lossFunctionGenerator(
          generatedSamples, realSamples
      )
      totalLossGenerator += lossGenerator.item()

    return totalLossGenerator / len(testLoader)

  def predict(self, df: DataFrame) -> None:
    """
    Predict the labels of the test data

    :param predict_df: prediction dataframe

    :return: predictions
    """
    raise NotImplementedError
    pass

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
      "lossFunctionDiscriminator": nn.BCELoss(),
      "lossFunctionGenerator": nn.MSELoss(),
      "downScaleFactor": 10,
      "noise": False,
    }
    return defaultParams
