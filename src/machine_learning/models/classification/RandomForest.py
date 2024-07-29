from typing import Literal
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from .IClassification import IClassification

class RandomForest(IClassification):
  """
  Random Forest classification model.
  """

  def __init__(
    self,
    labelName: Literal['shape', 'sample'] = 'shape',
    noise: bool = False,
    oneHotEncode: bool = False,
    params: dict = {}
  ):
    """
    Initializes a RandomForest object.

    Args:
      labelName: The name of the label column. Defaults to 'shape'.
      noise: Whether to add noise to the data. Defaults to False.
      oneHotEncode: Whether to perform one-hot encoding on the data. Defaults to False.
      params: Additional parameters for the RandomForestClassifier. Defaults to {}.
    """
    self.validParams = {
      'n_estimators': list(range(1, 1001)),
      'criterion': ['gini', 'entropy', 'log_loss'],
      'max_depth': list(range(1, 101)),
    }
    super().__init__(labelName, noise, oneHotEncode)
    self.model = RandomForestClassifier(**params)

  def train(self, df: DataFrame) -> None:
    """
    Trains the model.

    Args:
      df: The input DataFrame containing the training data.
    """
    values, labels = super().getValuesAndLabels(df)

    self.model.fit(values, labels)

  def test(self, df: DataFrame) -> float:
    """
    Tests the RandomForest model.

    Args:
      df: The input DataFrame containing the test data.

    Returns:
      The accuracy score of the model on the test data.
    """
    values, labels = super().getValuesAndLabels(df)

    score = self.model.score(values, labels)

    return score

  def predict(self, df: DataFrame) -> list:
    """
    Makes predictions using the RandomForest model.

    Args:
      df: The input DataFrame containing the data to be predicted.

    Returns:
      The predicted labels and the actual labels.
    """
    values, labels = super().getValuesAndLabels(df)

    predictions = self.model.predict(values)
    return predictions, labels

  def getDefaultParams(self) -> dict:
    """
    Returns the default parameters of the RandomForest model.

    Returns:
      The default parameters of the model.
    """
    return self.model.get_params()
