from typing import Literal

from sklearn.model_selection import GridSearchCV

from models.KNearestNeighbors import KNearestNeighbors

import pandas as pd


def runModels():
  """
  Run all models on the data
  """
  df_train, df_test, df_val = getData()

  models = [KNearestNeighbors()]
  params = {
      'n_neighbors': list(range(1, 22, 2)),
      'weights': ['uniform', 'distance'],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
  }
  for model in models:
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
  df = pd.read_csv(f"./data/data_samples_{material}.csv")
  df_train = df[df['set'] == 'train']
  df_test = df[df['set'] == 'test']
  df_val = df[df['set'] == 'val']

  return df_train, df_test, df_val

if __name__ == "__main__":
  runModels()
