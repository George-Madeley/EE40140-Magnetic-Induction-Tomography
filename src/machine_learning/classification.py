import os
from typing import Literal

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from models.classification import *

import pandas as pd


def runModels():
  '''
  Run all models on the data
  '''
  params = {
      'activation': ['identity', 'logistic', 'tanh', 'relu'],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
      'alpha': [0.0001, 0.001, 0.01, 0.1],
      'C': [0.1, 1, 10, 100],
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
        KNN(noise=noise),
        DT(noise=noise),
        NC(noise=noise),
        RF(noise=noise),
        SGD(noise=noise),
        SVM(noise=noise),
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

def getPredictionsPerLabel():

  materials = ['iron', 'copper']
  labels = ['shape', 'sample']

  params = {
      'activation': ['identity', 'logistic', 'tanh', 'relu'],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
      'alpha': [0.0001, 0.001, 0.01, 0.1],
      'C': [0.1, 1, 10, 100],
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

  models = [
    {
      'model': KNearestNeighbors,
      'params': {
        'n_neighbors': 3,
        'weights': 'distance',
        'metric': 'manhattan',
        'algorithm': 'auto'
      }
    },
    {
      'model': DecisionTree,
      'params': {
        'criterion': 'gini',
        'splitter': 'best',
        'max_depth': 901
      }
    },
    {
      'model': NearestCentroid,
      'params': {
        'metric': 'chebyshev'
      }
    },
    {
      'model': RandomForest,
      'params': {
        'n_estimators': 1,
        'criterion': 'log_loss',
        'max_depth': 401
      }
    },
    {
      'model': StochasticGradientDescent,
      'params': {
        'epsilon': 0.41,
        'loss': 'modified_huber',
        'max_iter': 400,
        'penalty': 'elasticnet'
      }
    },
    {
      'model': SupportVectorMachine,
      'params': {
        'cache_size': 200,
        'degree': 5,
        'kernel': 'poly'
      }
    },
  ]

  
  for material, label in list(zip(materials, labels)):
    print(f'Running models on {material}')
    df_train, df_test, df_val = getData(material=material)
    for noise in [True, False]:
      print(f'Running models with noise={noise}')
      for model in models:
        modelInstance = model['model'](
          noise=True,
          labelName=label,
          params=model['params']
        )

        modelInstance.train(df_train)

        predictions, labels = modelInstance.predict(df_val)

        # create a confusion matrix for the predictions and labels
        cm = confusion_matrix(labels, predictions)

        # export the confusion matrix to a csv file
        cmFilePath = os.path.join('results', 'classification', f'{modelInstance.__class__.__name__}-{material}-{noise}-confusion_matrix.csv')
        pd.DataFrame(cm).to_csv(cmFilePath, index=False)


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
  getPredictionsPerLabel()
