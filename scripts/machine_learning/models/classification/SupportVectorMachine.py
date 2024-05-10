from typing import List, Literal, Union

from sklearn.svm import SVC
from pandas import DataFrame

from .IClassification import IClassification

class SupportVectorMachine(IClassification):
  """
  SupportVectorMachine class for performing classification using Support Vector
  Machine algorithm.
  """
  def __init__(
      self,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = False,
      oneHotEncode: bool = False,
      params: dict = {}
    ):
    """
    Initializes a SupportVectorMachine object.

    Args:
      labelName: The name of the label column. Defaults to 'shape'.
      noise: Whether to add noise to the data. Defaults to False.
      oneHotEncode: Whether to perform one-hot encoding on the data. Defaults to False.
      params: Additional parameters for the RandomForestClassifier. Defaults to {}.
    """
    self.validParams = {
      'C': [0.1, 1, 10, 100],
      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
      'degree': list(range(1, 6)),
      'cache_size': [200, 400, 600, 800, 1000],
    }
    super().__init__(labelName, noise, oneHotEncode)
    self.model = SVC(**params)
  
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
    Tests the model.

    Args:
      df: The input DataFrame containing the test data.

    Returns:
      The accuracy score of the model on the test data.
    """
    values, labels = super().getValuesAndLabels(df)

    score = self.model.score(values, labels)

    return score
  
  def predict(self, df: DataFrame) -> List[Union[float, List[float]]]:
    """
    Makes predictions using the model.

    Args:
      df: The input DataFrame containing the data to be predicted.

    Returns:
      The predicted labels and the actual labels.
    """
    values, labels = super().getValuesAndLabels(df)
    predictions = self.model.predict(values)

    return predictions, labels
  
  def getDefaultParams(self):
    """
    Returns the default parameters of the model.

    Returns:
      The default parameters of the model.
    """
    return self.model.get_params()