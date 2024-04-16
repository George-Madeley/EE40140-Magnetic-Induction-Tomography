from typing import Literal
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

from ..IModel import IModel


class RandomForest(IModel):
  def __init__(
    self,
    labelName: Literal['shape', 'sample'] = 'shape',
    noise: bool = False,
    oneHotEncode: bool = False
  ):
    """
    Initializes a RandomForest object.

    Parameters:
    - labelName: The name of the label to predict. Can be either 'shape' or 'sample'. Default is 'shape'.
    - noise: Whether to add noise to the data. Default is False.
    """
    self.validParams = {
      'n_estimators': list(range(1, 1001)),
      'criterion': ['gini', 'entropy', 'log_loss'],
      'max_depth': list(range(1, 101)),
    }
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.model = RandomForestClassifier()

  def train(self, df: DataFrame) -> None:
    """
    Trains the random forest model using the provided DataFrame.

    Args:
      df (DataFrame): The input DataFrame containing the training data.

    Returns:
      None
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    self.randomForest.fit(values, labels)

  def test(self, df: DataFrame) -> float:
    """
    Test the random forest model on the given DataFrame.

    Args:
      df (DataFrame): The DataFrame containing the test data.

    Returns:
      float: The score of the random forest model on the test data.
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    score = self.randomForest.score(values, labels)

    return score
  
  def predict(self, df: DataFrame) -> list:
    """
    Predicts the labels for the given DataFrame using the trained random forest model.

    Args:
      df (DataFrame): The input DataFrame containing the features.

    Returns:
      list: The predicted labels for the input DataFrame.
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    return self.randomForest.predict(values)
  
  def getDefaultParams(self) -> dict:
    """
    Get the default parameters

    :return: default parameters
    """
    return self.model.get_params()
    