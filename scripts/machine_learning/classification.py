import os
from typing import Literal

from sklearn.model_selection import GridSearchCV

from models import *

import pandas as pd


def runModels():
  '''
  Run all models on the data
  '''
  params = {
      'activation': ['identity', 'logistic', 'tanh', 'relu'],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
      'alpha': [0.0001, 0.001, 0.01, 0.1],
      # 'C': [0.1, 1, 10, 100],
      'cache_size': [200, 400, 600, 800, 1000],
      "criterion": ["gini", "entropy", "log_loss"],
      'degree': list(range(1, 6)),
      'epsilon': [x / 100 for x in range(1, 101, 10)],
      'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
      'learning_rate': ['constant', 'invscaling', 'adaptive'],
      'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
      "max_depth": list(range(1, 1001, 100)),
      'max_iter': list(range(100, 1001, 100)),
      'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
      'n_estimators': list(range(1, 1001, 100)),
      'n_neighbors': list(range(3, 31, 3)),
      'penalty': ['l2', 'l1', 'elasticnet'],
      "splitter": ["best", "random"],
      'weights': ['uniform', 'distance'],
  }

  
  for material in ['iron', 'copper']:
    print(f'Running models on {material}')
    df_train, df_test, df_val = getData(material=material)
    for noise in [True, False]:
      print(f'Running models with noise={noise}')
      models = [
        # KNearestNeighbors(noise=noise),
        # DecisionTree(noise=noise),
        # NearestCentroid(noise=noise),
        # RandomForest(noise=noise),
        NeuralNetwork(noise=True),
        # StochasticGradientDescent(noise=noise),
        # SupportVectorMachine(noise=noise),
      ]
      for model in models:
        scoring = {
          'Accuracy': 'accuracy',
          'F1': 'f1_micro',
          'Precision': 'precision_micro',
          'Recall': 'recall_micro',
        }

        if (
          model.__class__.__name__ == 'NeuralNetwork' or
          model.__class__.__name__ == 'KNearestNeighbors' or
          model.__class__.__name__ == 'DecisionTree' or
          model.__class__.__name__ == 'RandomForest'
          ):
          scoring['AUC'] = 'roc_auc_ovr'

        print(f'Running {model.__class__.__name__}')
        model.varyParams(
            df_train,
            params,
            searchCV=GridSearchCV,
            verbose=2,
            material=material,
            scoring=scoring,
            numSamples=240 if noise else 120
        )


def getData(material: Literal['iron', 'copper'] = 'iron'):
  '''
  Get the data for the specified material

  :param material: the material to get the data for

  :return: the data
  '''
  dataFilePath = os.path.join('data', f'data_samples_{material}.csv')
  df = pd.read_csv(dataFilePath)
  df_train = df[df['set'] == 'train']
  df_test = df[df['set'] == 'test']
  df_val = df[df['set'] == 'val']

  return df_train, df_test, df_val

if __name__ == '__main__':
  runModels()
