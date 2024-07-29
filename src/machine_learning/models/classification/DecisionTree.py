from typing import Literal, Tuple
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier

from .IClassification import IClassification

class DecisionTree(IClassification):
  """
  DecisionTree class for performing classification using Decision Tree algorithm.
  """

  def __init__(
    self,
    labelName: Literal['shape', 'sample'] = 'shape',
    noise: bool = False,
    oneHotEncode: bool = False,
    params: dict = {}
  ):
    """
    Initializes a Decision Tree object.

    Args:
      labelName: The name of the label column. Defaults to 'shape'.
      noise: Whether to add noise to the data. Defaults to False.
      oneHotEncode: Whether to perform one-hot encoding on the data. Defaults to False.
      params: Additional parameters for the RandomForestClassifier. Defaults to {}.
    """
    self.validParams = {
      "criterion": ["gini", "entropy", "log_loss"],
      "splitter": ["best", "random"],
      "max_depth": list(range(1, 1001)),
    }
    super().__init__(labelName, noise, oneHotEncode)
    self.model = DecisionTreeClassifier(**params)

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
  
  def predict(self, df: DataFrame) -> Tuple[list, list]:
    """
    Makes predictions using the model.

    Args:
      df: The input DataFrame containing the data to be predicted.

    Returns:
      The predicted labels and the actual labels.
    """
    values, labels = super().getValuesAndLabels(df)

    return self.model.predict(values), labels
  
  def getDefaultParams(self) -> dict:
    """
    Returns the default parameters of the model.

    Returns:
      The default parameters of the model.
    """
    return self.model.get_params()
  

