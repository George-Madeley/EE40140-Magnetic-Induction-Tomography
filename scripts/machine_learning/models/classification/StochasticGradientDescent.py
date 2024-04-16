from typing import List, Literal, Union

from sklearn.linear_model import SGDClassifier
from pandas import DataFrame

from ..IModel import IModel

class StochasticGradientDescent(IModel):
  def __init__(
      self,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = False,
      oneHotEncode: bool = False
    ):
    """
    Initializes a StochasticGradientDescent object.

    Returns:
    - None
    """
    self.validParams = {
      'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
      'penalty': ['l2', 'l1', 'elasticnet'],
      'alpha': [x / 100 for x in range(1, 101)],
      'l1_ratio': [x / 100 for x in range(1, 101)],
      'fit_intercept': [True, False],
      'max_iter': list(range(1, 1001)),
      'tol': [x / 100 for x in range(1, 101)],
      'shuffle': [True, False],
      'epsilon': [x / 100 for x in range(1, 101)],
      'n_jobs': [-1, None],
    }
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.model = SGDClassifier()

  def train(self, df: DataFrame) -> None:
    """
    Train the model

    :param train_df: training dataframe
    :param noise: whether to include background noise
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    self.model.fit(values, labels)

  def test(self, df: DataFrame) -> float:
    """
    Test the model

    :param test_df: test dataframe
    :param noise: whether to include background noise

    :return: score
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    score = self.model.score(values, labels)

    return score
  
  def predict(self, df: DataFrame) -> List[Union[float, List[float]]]:
    """
    Predict the labels of the given data

    :param df: the data to predict

    :return: the predicted labels
    """
    values, _ = super().getValuesAndLabels(df, self.labelName, self.noise)

    return self.model.predict(values)
  
  def predictProba(self, df: DataFrame) -> List[List[float]]:
    """
    Predict the probabilities of the given data

    :param df: the data to predict

    :return: the predicted probabilities
    """
    values, _ = super().getValuesAndLabels(df, self.labelName, self.noise)

    return self.model.predict_proba(values)
  
  def getDefaultParams(self) -> dict:
    """
    Get the default parameters

    :return: default parameters
    """
    return self.model.get_params()