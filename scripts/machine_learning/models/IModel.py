from abc import ABC, abstractmethod
from random import choice
from string import ascii_letters
from sys import _getframe
from typing import Literal, get_args, get_origin

from pandas import DataFrame, concat
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class IModel(ABC):
  @abstractmethod
  def train(self, train_df):
    """
    Train the model

    :param train_df: training dataframe
    """
    pass

  @abstractmethod
  def test(self, test_df):
    """
    Test the model

    :param test_df: test dataframe

    :return: metrics
    """
    pass

  @abstractmethod
  def predict(self, predict_df):
    """
    Predict the labels of the test data

    :param predict_df: prediction dataframe

    :return: predictions
    """
    pass

  @abstractmethod
  def getDefaultParams(self):
    """
    Get the default parameters

    :return: default parameters
    """
    pass

  @staticmethod
  def enforceLiterals(function):
    kwargs = _getframe(1).f_locals
    for name, type_ in function.__annotations__.items():
      value = kwargs.get(name)
      options = get_args(type_)
      if get_origin(
        type_) is Literal and name in kwargs and value not in options:
        raise AssertionError(f"'{value}' is not in {options} for '{name}'")
      
  def varyParams(
      self,
      df: DataFrame,
      params: dict,
      searchCV: type[GridSearchCV | RandomizedSearchCV],
      scoring: dict = None,
      n_jobs: int = -1,
      verbose: int = 0
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
        'Precision': 'precision_micro',
        'Recall': 'recall_micro',
        'Accuracy': 'accuracy',
        'F1': 'f1_micro',
        'AUC': 'roc_auc',
      }

    values, labels = self.getValuesAndLabels(df, self.labelName, self.noise)

    clf = searchCV(
      self.model,
      validParams,
      scoring=scoring,
      n_jobs=n_jobs,
      verbose=verbose,
      refit=False
    )
    clf.fit(values, labels)


    df_results = concat(
      [
        DataFrame(clf.cv_results_['params'])
      ] + [
        DataFrame(
          clf.cv_results_[f'mean_test_{metric}'],
          columns=[metric]
        ) for metric in scoring.keys()
      ],
      axis=1
    )

    # Add the column 'model' to the dataframe and set it to the model name
    df_results.insert(0, 'model', self.__class__.__name__)

    # Generate a random string
    randomString = ''.join(
      [choice(ascii_letters) for i in range(10)]
    )

    # Save the results to a CSV file
    df_results.to_csv(
      f'./results/{self.__class__.__name__}_results_{randomString}.csv',
      index=False
    )

  def isParamValid(self, name, value=None):
    """
    Check if the parameter is valid

    :param name: name of the parameter
    :param value: value of the parameter

    :return: boolean
    """
    if name in self.validParams.keys():
      return True
    return False

  def getValuesAndLabels(
    self,
    df,
    labelName: Literal['shape', 'sample'] = 'shape',
    noise: bool = False
  ):
    """
    Get the values and labels from the dataframe

    :param df: dataframe
    :param labelName: label column name

    :return: values, labels
    """

    valueColumnNames = [col for col in df.columns if col.startswith('cc_')]
    valueColumnNames.remove('cc_filename')
    if noise:
      valueColumnNames += [col for col in df.columns if col.startswith('bb_')]
      valueColumnNames.remove('bb_filename')

    values = df[valueColumnNames].values

    labelColumnNames = [col for col in df.columns if col.startswith(labelName + '_')]
    labels = df[labelColumnNames].values

    return values, labels
