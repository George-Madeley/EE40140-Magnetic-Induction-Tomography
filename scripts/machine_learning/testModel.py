import os
from typing import Literal

from sklearn.model_selection import GridSearchCV

from models import *

import pandas as pd


def runModels():
  """
  Run all models on the data
  """
  df_train, df_test, df_val = getData()

  models = [
    StochasticGradientDescent(),
  ]
  params = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
  }
  for model in models:
    print(f"Running {model.__class__.__name__}")
    model.varyParams(
        df_train,
        params,
        searchCV=GridSearchCV,
        verbose=2,
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

if __name__ == "__main__":
  runModels()
