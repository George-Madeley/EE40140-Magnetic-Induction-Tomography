import csv
import os
from random import choice
from string import ascii_letters
from typing import List, Literal


from models import GAN, VAE
import pandas as pd

def main():
  """
  Run the generation models
  """
  df_train, df_test, df_val = getData()

  batchSize = 45

  commonFactors = getCommonFactors(len(df_train), len(df_test))
  if batchSize not in commonFactors:
    raise ValueError(
      f"Batch size must be a common factor of the number of training and testing samples. Common factors are: {commonFactors}"
    )

  modelClasses = [
    GAN
  ]

  params = {
    'downScaleFactor': [16, 10, 8, 5, 4, 2, 1],
    'learningRate': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
  }

  for modelClass in modelClasses:
    varyParams(modelClass, df_train, params)

  pass

def varyParams(modelClass, df_train, params):
  """
  Vary the parameters of the models
  """

  dfpath = os.path.join(
    'results',
    f'{modelClass.__name__}.csv'
  )

  if not os.path.exists(dfpath):
    with open(dfpath, 'w', newline='') as f:
      headers = ['Model Name'] + list(params.keys()) + [
        'Epoch',
        'D Loss',
        'G Loss',
        'Img ID',
      ]

      writer = csv.DictWriter(f, fieldnames=headers)
      writer.writeheader()


  defaultParams = modelClass.getDefaultParams()
    
  for param in params.keys():
    for paramValue in params.get(param, []):
      uniqueId = ''.join([choice(ascii_letters) for i in range(10)])
      imageDir = os.path.join('images', 'epochs', f'{modelClass.__name__}', f'{uniqueId}')
      os.makedirs(imageDir, exist_ok=True)

      defaultParams[param] = paramValue

      model = modelClass(
        **defaultParams
      )
      model.train(
        df_train,
        imageDir,
        dfpath
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
  main()