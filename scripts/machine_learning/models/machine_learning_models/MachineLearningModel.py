from sys import _getframe
from typing import Literal, get_args, get_origin

import pandas as pd

class MachineLearningModel:
  @staticmethod
  def enforceLiterals(function):
    kwargs = _getframe(1).f_locals
    for name, type_ in function.__annotations__.items():
      value = kwargs.get(name)
      options = get_args(type_)
      if get_origin(type_) is Literal and name in kwargs and value not in options:
        raise AssertionError(f"'{value}' is not in {options} for '{name}'")

  def getLabels(self, df):
    """
    Get the labels from the dataframe
    
    :param df: dataframe
    
    :return: labels
    """
    return df['shape'].values
  
  def getValuesWithBackgroundNoise(self, df):
    """
    Get the values with background noise
    
    :param df: dataframe
    
    :return: values
    """
    # Get the values of the background noise
    bbColumnNames = df.filter(regex='^bb_\d{1,3}$').columns
    bbValues = df[bbColumnNames].values

    # Get the values of the sample
    ccColumnNames = df.filter(regex='^cc_\d{1,3}$').columns
    ccValues = df[ccColumnNames].values

    # Combine the values of the background noise and the sample to the new
    # dataframe has a total of 240 columns
    values = pd.concat([bbValues, ccValues], axis=1)

    return values
  
  def getValuesWithoutBackgroundNoise(self, df):
    """
    Get the values without background noise
    
    :param df: dataframe

    :return: values
    """

    # Get the values of the sample
    ccColumnNames = df.filter(regex='^cc_\d{1,3}$').columns
    ccValues = df[ccColumnNames].values

    return ccValues