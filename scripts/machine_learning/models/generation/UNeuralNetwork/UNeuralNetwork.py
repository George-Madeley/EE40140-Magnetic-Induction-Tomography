from torch import nn
from torch.optim import Adam

from .Contractor import Contractor
from .Expandor import Expandor
from ..IGeneration import IGeneration


class UNeuralNetwork(IGeneration):
  def __init__(self, structure, **kwargs):
    super().__init__(**kwargs)

    contractorStructure = structure.get('contractor')
    expandorStructure = structure.get('expandor')

    self.contractor = Contractor(
      contractorStructure
    ).to(self.device)
    self.expandor = Expandor(
      640 // self.downScaleFactor,
      480 // self.downScaleFactor,
      expandorStructure
    ).to(self.device)

    self.optimizerContractor = Adam(
      self.contractor.parameters(),
      lr=self.learningRate
    )
    self.optimizerExpandor = Adam(
      self.expandor.parameters(),
      lr=self.learningRate
    )

    self.eLossFunc = nn.MSELoss()

    self.modelNames = ['UNET']

  def train(
    self,
    trainLoader,
    metrics: list[str]
  ) -> None:
    trainLoaderLen = len(trainLoader)

    for realImagesSamples, signalSamples in trainLoader:
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      self.optimizerContractor.zero_grad()
      self.optimizerExpandor.zero_grad()

      latentSamples = self.contractor(signalSamples)
      generatedImageSamples = self.expandor(latentSamples)

      loss = self.eLossFunc(realImagesSamples, generatedImageSamples)
      loss.backward()
      self.optimizerContractor.step()
      self.optimizerExpandor.step()

      losses = super().score(metrics, realImagesSamples, generatedImageSamples)

    losses = {f'UNET {k}': v / trainLoaderLen for k, v in losses.items()}

    return loss, losses

  def test(
      self,
      testLoader,
      metrics: dict,
    ) -> float:
    """
    Test the generative adversarial network model on the given DataFrame and return the accuracy score.

    Parameters:
    - df (DataFrame): The DataFrame containing the test data.

    Returns:
    - float: The accuracy score of the generative adversarial network model on the test data.
    """
    testLoaderLen = len(testLoader)

    for realImagesSamples, signalSamples in testLoader:
      signalSamples = signalSamples.to(self.device)
      realImagesSamples = realImagesSamples.to(self.device)

      latentSamples = self.contractor(signalSamples)
      generatedImageSamples = self.expandor(latentSamples)

      losses = super().score(metrics, realImagesSamples, generatedImageSamples)
    
    losses = {f'UNET {k}': v / testLoaderLen for k, v in losses.items()}
    return losses

  def predict(self, fixedSignalSamples, imgDir, epoch) -> None:
    """
    Predict the labels of the test data

    :param predict_df: prediction dataframe

    :return: predictions
    """
    fixedLatentSamples = self.contractor(fixedSignalSamples)
    fixedGeneratedImages = self.expandor(fixedLatentSamples)
    fixedGeneratedImages = fixedGeneratedImages.detach().cpu()
    self.plotImages(imgDir, fixedGeneratedImages, f'epoch_{str(epoch).zfill(3)}.png', subtitle=f'Epoch {epoch}')
