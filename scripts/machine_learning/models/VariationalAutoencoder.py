import os
from typing import Literal

from matplotlib import pyplot as plt
from pandas import DataFrame

from .IModel import IModel
from .VAE import VAE

import torch
from torch.optim import Adam
from torch import nn

class VariationalAutoencoder(IModel):
  def __init__(
    self,
    labelName: Literal['shape', 'sample'] = 'shape',
    noise: bool = False,
    oneHotEncode: bool = False,
    downScaleFactor: int = 1,
    **kwargs
  ) -> None:
    
    self.originalWidth = 640
    self.originalHeight = 480
    
    self.learningRate = kwargs.get("learningRate", 0.0002)
    self.numEpochs = kwargs.get("numEpochs", 1000)
    self.batchSize = kwargs.get("batchSize", 64)

    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.downScaleFactor = downScaleFactor

    
    self.device = ""
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")


    inputDim = 240 if self.noise else 120
    hiddenDim = kwargs.get("hiddenDim", 400)
    latentDim = kwargs.get("latentDim", 200)

    self.model = VAE(
      device=self.device,
      input_dim=inputDim,
      hidden_dim=hiddenDim,
      latent_dim=latentDim,
      height=self.originalHeight // downScaleFactor,
      width=self.originalWidth // downScaleFactor
    ).to(self.device)

  def lossFunction(self, x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
  
  def train(self, df:DataFrame) -> None:
    values, _ = self.getValuesAndLabels(df)
    realImages = self.getImages(df, self.downScaleFactor)

    values = torch.from_numpy(values).float()

    trainSet = torch.utils.data.TensorDataset(realImages, values)

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
    savePath = os.path.join('images', 'epochs', f'original.png')
    plt.savefig(savePath)

    optimizer = Adam(
      self.model.parameters(),
      lr=self.learningRate
    )
    
    self.model.train()
    for epoch in range(self.numEpochs):
      overall_loss = 0
      for n, (realImages, signal) in enumerate(trainLoader):
        realImages = realImages.to(device=self.device)
        signal = signal.to(self.device)

        optimizer.zero_grad()

        x_hat, mean, log_var = self.model(signal)
        loss = self.lossFunction(realImages, x_hat, mean, log_var)

        overall_loss += loss.item()

        loss.backward()
        optimizer.step()

      print(f"Epoch: {epoch} Loss VAE.: {overall_loss / (n*numBatches)}")
      print("--------------------------------------------------")

      # Generate images using random latent samples
      if epoch == 0:
        fixedLatentSamples = trainLoader.dataset.tensors[1][randomSampleIndeces].to(self.device)
      generatedImages = self.model(fixedLatentSamples)[0]
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
      savePath = os.path.join('images', 'epochs', f'epoch_{str(epoch).zfill(3)}.png')
      plt.savefig(savePath)

  def test(self, df: DataFrame) -> None:
    """
    Test the model

    :param test_df: test dataframe

    :return: None
    """
    values, _ = self.getValuesAndLabels(df)
    realImages = self.getImages(df, self.downScaleFactor)
    values = torch.from_numpy(values).float()
    testSet = torch.utils.data.TensorDataset(realImages, values)
    testLoader = torch.utils.data.DataLoader(
        testSet, batch_size=self.batchSize, shuffle=True
    )
    
    totalLoss = 0
    for realImages, signal in testLoader:
      realImages = realImages.to(device=self.device)
      signal = signal.to(self.device)

      x_hat, mean, log_var = self.model(signal)
      loss = self.lossFunction(realImages, x_hat, log_var)

      totalLoss += loss.item()

    return totalLoss / len(testLoader)


  def predict(self, df: DataFrame) -> None:
    """
    Predict the labels of the test data

    :param predict_df: prediction dataframe

    :return: predictions
    """
    raise NotImplementedError
    pass

  def getDefaultParams(self):
    """
    Get the default parameters

    :return: default parameters
    """
    raise NotImplementedError
    pass
