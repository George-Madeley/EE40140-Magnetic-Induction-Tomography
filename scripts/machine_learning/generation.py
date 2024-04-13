import os
from typing import List, Literal

import pandas as pd

from models import GAN, VAE

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
    VAE
  ]

  for modelClass in modelClasses:
    model = modelClass(
      downScaleFactor=20,
      batchSize=batchSize
    )
    model.train(df_train)

  pass

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