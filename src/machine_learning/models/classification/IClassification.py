import os
from abc import abstractmethod
from random import choice
from string import ascii_letters
from typing import Literal

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from pandas import DataFrame, concat

from ..IModel import IModel

class IClassification(IModel):
  def __init__(
      self,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = False,
      oneHotEncode: bool = False
    ):
    """
    Initializes a IClassification object.

    :param labelName: The name of the label column. Defaults to 'shape'.
    :param noise: Whether to add noise to the data. Defaults to False.
    :param oneHotEncode: Whether to perform one-hot encoding on the data. Defaults to False.
    """
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode


  @abstractmethod
  def getDefaultParams(self):
    """
    Get the default parameters

    :return: default parameters
    """
    pass

  def varyParams(
      self,
      df: DataFrame,
      params: dict,
      searchCV: type[GridSearchCV | RandomizedSearchCV],
      scoring: dict = None,
      n_jobs: int = -1,
      verbose: int = 0,
      material: Literal['iron', 'copper'] = 'iron',
      numSamples: Literal[120, 240] = 120,
    ) -> None:
    """
    Vary the parameters of the model

    :param params: parameters
    """
    validParams = {}
    for param in params.keys():
      if self.isParamValid(param):
        validParams[param] = params[param]

    if validParams == {}:
      print(
        f'No valid parameters to vary for for {self.__class__.__name__} model.')

    if scoring is None:
      scoring = {
        'Accuracy': 'accuracy',
        'F1': 'f1_micro',
        'Precision': 'precision_micro',
        'Recall': 'recall_micro',
      }

    values, labels = self.getValuesAndLabels(df)

    clf = searchCV(
      self.model,
      validParams,
      scoring=scoring,
      n_jobs=n_jobs,
      verbose=verbose,
      refit=False
    )
    clf.fit(values, labels)

    df_results = concat([
      DataFrame(clf.cv_results_['params']),
      DataFrame({metric: clf.cv_results_[f'mean_test_{metric}'] for metric in scoring.keys()}),
      DataFrame(clf.cv_results_['mean_fit_time'], columns=['mean_fit_time']),
      DataFrame(clf.cv_results_['std_fit_time'], columns=['std_fit_time']),
      DataFrame(clf.cv_results_['mean_score_time'], columns=['mean_score_time']),
      DataFrame(clf.cv_results_['std_score_time'], columns=['std_score_time'])
    ], axis=1)

    # Add the column 'model' to the dataframe and set it to the model name
    df_results.insert(0, 'model', self.__class__.__name__)
    df_results.insert(1, 'material', material)
    df_results.insert(2, 'numSamples', numSamples)

    # Generate a random string
    randomString = ''.join(
      [choice(ascii_letters) for i in range(10)]
    )

    saveFilePath = os.path.join(
        'results', f'{self.__class__.__name__}_results_{randomString}.csv')
    # Save the results to a CSV file
    df_results.to_csv(
      saveFilePath,
      index=False
    )

  def isParamValid(self, name: str, value=None):
    """
    Check if the parameter is valid

    :param name: name of the parameter
    :param value: value of the parameter

    :return: boolean
    """
    if name in self.validParams.keys():
      return True
    return False