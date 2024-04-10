from random import choice
from string import ascii_letters
from typing import List, Union

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame, concat

from IModel import IModel

class KNearestNeighbors(IModel):
  def __init__(self, k: int, weights: str = 'distance'):
    """
    Initializes a KNearestNeighbors object.

    Parameters:
    - k (int): The number of nearest neighbors to consider.
    - weights (str): The weight function used in prediction. Default is 'distance'.

    Returns:
    - None
    """
    self.model = self.createModel(k, weights)

  def createModel(self, k: int, weights: str = 'distance') -> KNeighborsClassifier:
    """
    Create a k nearest neighbor model

    :param k: number of neighbors
    :param weights: weight function used in prediction

    :return: model
    """
    model = KNeighborsClassifier(n_neighbors=k, weights=weights)
    return model

  def train(self, df_train: DataFrame, noise: bool = False) -> None:
    """
    Train the model

    :param train_df: training dataframe
    :param noise: whether to include background noise
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_train)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_train)

    labels = super().getLabels(df_train)

    self.model.fit(values, labels)

  def test(self, df_test: DataFrame, noise: bool = False) -> float:
    """
    Test the model

    :param test_df: test dataframe
    :param noise: whether to include background noise

    :return: score
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_test)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_test)

    labels = super().getLabels(df_test)

    score = self.model.score(values, labels)

    return score

  def predict(self, df_predictions: DataFrame, noise: bool = False) -> List[Union[float, List[float]]]:
    """
    Predict the labels of the test data

    :param test_df: test dataframe
    :param noise: whether to include background noise

    :return: predictions
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_predictions)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_predictions)

    labels = super().getLabels(df_predictions)

    predictions = self.model.predict_proba(values)

    return predictions, labels
  
  def varyParams(
      self,
      df: DataFrame,
      params: dict,
      searchCV: type[GridSearchCV | RandomizedSearchCV],
      noise: bool = False,
      scoring: dict = None,
      n_jobs: int = -1,
      verbose: int = 0
    ) -> None:
    """
    Vary the parameters of the model

    :param params: parameters
    """
    for param in params.keys():
      if not self.isParamValid(param, params[param]):
        # Remove the parameter from the dictionary if it is invalid
        del params[param]

    if params == {}:
      print(f'No valid parameters to vary for for {self.__class__.__name__} model.')
      
    if scoring is None:
      scoring = {
        'Precision': 'precision',
        'Recall': 'recall',
        'Accuracy': 'accuracy',
        'F1': 'f1',
        'AUC': 'roc_auc',
      }

    if noise:
      values = super().getValuesWithBackgroundNoise(df)
    else:
      values = super().getValuesWithoutBackgroundNoise(df)

    labels = super().getLabels(df)
    
    clf = searchCV(
      self.model,
      params,
      scoring=scoring,
      n_jobs=n_jobs,
      verbose=verbose
    )
    clf.fit(values, labels)

    df_results = concat(
      [
        DataFrame(clf.cv_results_['params']),
        DataFrame(
          clf.cv_results_['mean_test_score'],
          columns=['mean_test_score']
        ),
      ],
      axis=1
    )

    # Generate a random string
    randomString = ''.join(
      [choice(ascii_letters) for i in range(10)]
    )

    # Save the results to a CSV file
    df_results.to_csv(
      f'{self.__class__.__name__}_results_{randomString}.csv',
      index=False
    )

  
  @staticmethod
  def isParamValid(self, name: str, value: str) -> bool:
    """
    Check if the parameter is valid

    :param name: name of the parameter
    :param value: value of the parameter

    :return: boolean
    """
    validParams = {
      'weights': ['uniform', 'distance'],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
      'leaf_size': list(range(1, 101)),
      'p': [1, 2],
      'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
      'n_jobs': [-1, None],
      'n_neighbors': list(range(1, 101)),
    }

    if name in validParams.keys() and value in validParams.get(name, []):
      return True
    return False
  
  def getDefaultParams(self) -> dict:
    """
    Get the default parameters

    :return: default parameters
    """
    return self.model.get_params()
