import os
from typing import List, Literal


from models import VariationalAutoencoder as VAE
from models import GenerativeAdversarialNetwork as GAN
from models import GenerationNeuralNetwork as NN
import pandas as pd

def runModels():
  """
  Run the generation models
  """
  df_train, df_test, df_val = getData()

  structureNN = {
    'NN1': [480]
  }

  models = [
    # NN(
    #   labelName='shape',
    #   noise=True,
    #   downScaleFactor=8,
    #   structure=structureNN.get('NN1'),
    #   name='NN1',
    #   batchSize=16,
    #   learningRate=0.001,
    #   maxEpochs=100,
    # ),
    # GAN(
    #   labelName='shape',
    #   noise=True,
    #   downScaleFactor=8,
    #   name='GAN1',
    #   batchSize=45,
    #   learningRate=0.0001,
    #   maxEpochs=100,
    # )
    VAE(
      labelName='shape',
      noise=True,
      downScaleFactor=8,
      name='VAE1',
      batchSize=45,
      learningRate=0.0001,
      maxEpochs=100,
    )
  ]

  for model in models:
    print(f"Running {model.__class__.__name__}")
    model.run(
        df_train,
        df_test,
        df_val
    )
  

def getData(material: Literal['iron', 'copper'] = 'iron'):
  """
  Get the data for the specified material

  :param material: the material to get the data for

  :return: the data
  """
  dataFilePath = os.path.join('data', f'data_samples_{material}.csv')
  df = pd.read_csv(dataFilePath)
  df_train = df[df['set'] == 'train']
  df_test = df[df['set'] == 'test']
  df_val = df[df['set'] == 'val']

  return df_train, df_test, df_val

def getCommonFactors(a: int, b: int) -> List[int]:
  """
  Returns a list of common factors between two numbers within a specified range.

  Parameters:
  a (int): The first number.
  b (int): The second number.

  Returns:
  list: A list of common factors between a and b.

  """
  factors: List[int] = []
  for i in range(1, min(a, b) + 1):
    if a % i == 0 and b % i == 0:
      factors.append(i)
  return factors

if __name__ == "__main__":
  runModels()