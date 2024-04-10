from sklearn.neighbors import NearestCentroid as NearestCentroidClassifier
from IMLModel import IModel
from MachineLearningModel import MachineLearningModel
from pandas import DataFrame
from typing import List

class NearestCentroid(IModel, MachineLearningModel):
  def __init__(self):
    self.model = self.createModel()

  def createModel(self) -> NearestCentroidClassifier:
    """
    Create a nearest centroid model

    :return: model
    """
    model = NearestCentroidClassifier()
    return model

  def train(self, df_train: DataFrame, noise: bool = False) -> None:
    """
    Train the model

    :param df_train: training dataframe
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

    :param df_test: test dataframe
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

  def predict(self, df_predictions: DataFrame, noise: bool) -> List:
    """
    Predict the labels of the test data

    :param df_predictions: dataframe of predictions
    :param noise: whether to include background noise

    :return: predictions
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_predictions)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_predictions)

    predictions = self.model.predict(values)

    return predictions

  @staticmethod
  def isParamValid(self, name: str, value: str) -> bool:
    """
    Check if the parameter is valid

    :param name: name of the parameter
    :param value: value of the parameter

    :return: boolean
    """
    
    validParams = {
      'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
      'shrink_threshold': ['None']
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