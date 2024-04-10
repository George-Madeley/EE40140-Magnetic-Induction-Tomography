from abc import ABC, abstractmethod
from sys import _getframe
from typing import Literal, get_args, get_origin


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
  def varyParams(self, params):
    """
    Vary the parameters of the model

    :param params: parameters
    """
    pass

  @abstractmethod
  def isParamValid(self, name, value):
    """
    Check if the parameter is valid

    :param name: name of the parameter
    :param value: value of the parameter

    :return: boolean
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

    labelColumnNames = [col for col in df.columns if col.startswith(labelName)]
    labels = df[labelColumnNames].values

    return values, labels
