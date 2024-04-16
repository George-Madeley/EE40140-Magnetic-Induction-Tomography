from typing import List, Literal, Union

from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame

from ..IModel import IModel
class KNearestNeighbors(IModel):
  def __init__(
      self,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = False,
      oneHotEncode: bool = False
    ):
    """
    Initializes a KNearestNeighbors object.

    Returns:
    - None
    """
    self.validParams = {
      'weights': ['uniform', 'distance'],
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
      'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
      'n_neighbors': list(range(1, 101)),
    }
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.model = KNeighborsClassifier()

  def train(self, df: DataFrame) -> None:
    """
    Train the model

    :param train_df: training dataframe
    :param noise: whether to include background noise
    """
    values, labels = super().getValuesAndLabels(df, self.labelName. self.noise)

    self.model.fit(values, labels)

  def test(self, df: DataFrame) -> float:
    """
    Test the model

    :param test_df: test dataframe
    :param noise: whether to include background noise

    :return: score
    """
    values, labels = super().getValuesAndLabels(df, self.labelName. self.noise)

    score = self.model.score(values, labels)

    return score

  def predict(self, df: DataFrame) -> List[Union[float, List[float]]]:
    """
    Predict the labels of the test data

    :param test_df: test dataframe
    :param noise: whether to include background noise

    :return: predictions
    """
    values, labels = super().getValuesAndLabels(df, self.labelName. self.noise)

    predictions = self.model.predict_proba(values)

    return predictions, labels

  def getDefaultParams(self) -> dict:
    """
    Get the default parameters

    :return: default parameters
    """
    return self.model.get_params()
