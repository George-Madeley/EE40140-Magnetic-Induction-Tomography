from sklearn.neighbors import NearestCentroid as NearestCentroidClassifier

from pandas import DataFrame
from typing import List, Literal

from .IModel import IModel

class NearestCentroid(IModel):
  def __init__(
    self,
    labelName: Literal['shape', 'sample'] = 'shape',
    noise: bool = False
  ):
    """
    Initialize the NearestCentroid model.

    Args:
      labelName (Literal['shape', 'sample'], optional): The name of the label to use for classification. Defaults to 'shape'.
      noise (bool, optional): Whether to add noise to the data. Defaults to False.
    """
    self.validParams = {
      'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
      'shrink_threshold': ['None']
    }
    self.labelName = labelName
    self.noise = noise
    self.model = NearestCentroidClassifier()

  def train(self, df: DataFrame) -> None:
    """
    Trains the NearestCentroid model using the provided DataFrame.

    Args:
      df (DataFrame): The input DataFrame containing the training data.

    Returns:
      None
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    self.model.fit(values, labels)

  def test(self, df: DataFrame) -> float:
    """
    Test the model using the provided DataFrame.

    Args:
      df (DataFrame): The DataFrame containing the test data.

    Returns:
      float: The score of the model on the test data.
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    score = self.model.score(values, labels)

    return score

  def predict(self, df: DataFrame) -> List:
    """
    Predicts the labels for the given DataFrame using the trained model.

    Args:
      df (DataFrame): The input DataFrame containing the feature values.

    Returns:
      List: The predicted labels for the input DataFrame.
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    predictions = self.model.predict(values)

    return predictions
  
  def getDefaultParams(self) -> dict:
    """
    Get the default parameters

    :return: default parameters
    """
    return self.model.get_params()