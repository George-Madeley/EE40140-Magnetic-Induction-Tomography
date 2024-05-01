from typing import List, Literal, Union

from sklearn.linear_model import SGDClassifier
from pandas import DataFrame

from .IClassification import IClassification

class StochasticGradientDescent(IClassification):
  def __init__(
      self,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = False,
      oneHotEncode: bool = False,
      params: dict = {}
    ):
    """
    Initializes a StochasticGradientDescent object.

    Returns:
    - None
    """
    self.validParams = {
      'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
      'penalty': ['l2', 'l1', 'elasticnet'],
      'max_iter': list(range(1, 1001)),
      'epsilon': [x / 100 for x in range(1, 101)],
    }
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.model = SGDClassifier(**params)

  def train(self, df: DataFrame) -> None:
    """
    Train the model

    :param train_df: training dataframe
    :param noise: whether to include background noise
    """
    values, labels = super().getValuesAndLabels(df)

    self.model.fit(values, labels)

  def test(self, df: DataFrame) -> float:
    """
    Test the model

    :param test_df: test dataframe
    :param noise: whether to include background noise

    :return: score
    """
    values, labels = super().getValuesAndLabels(df)

    score = self.model.score(values, labels)

    return score
  
  def predict(self, df: DataFrame) -> List[Union[float, List[float]]]:
    """
    Predict the labels of the given data

    :param df: the data to predict

    :return: the predicted labels
    """
    values, labels = super().getValuesAndLabels(df)

    predictions = self.model.predict(values)
    return predictions, labels
  
  def getDefaultParams(self) -> dict:
    """
    Get the default parameters

    :return: default parameters
    """
    return self.model.get_params()