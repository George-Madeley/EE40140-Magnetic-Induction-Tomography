from typing import Literal
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier

from .IModel import IModel

class DecisionTree(IModel):
  def __init__(
    self,
    labelName: Literal['shape', 'sample'] = 'shape',
    noise: bool = False,
    oneHotEncode: bool = False
  ):
    """
    Initialize a DecisionTree object.

    Args:
        labelName (Literal['shape', 'sample'], optional): The name of the label to predict. Defaults to 'shape'.
        noise (bool, optional): Whether to add noise to the data. Defaults to False.
    """
    self.validParams = {
      "criterion": ["gini", "entropy", "log_loss"],
      "splitter": ["best", "random"],
      "max_depth": list(range(1, 1001)),
      "min_samples_split": list(range(1, 1001)),
      "min_samples_leaf": list(range(1, 1001)),
      "max_features": ["sqrt", "log2"],
      "max_leaf_nodes": list(range(1, 1001))
    }
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.model = DecisionTreeClassifier()

  def train(self, df: DataFrame) -> None:
    """
    Trains the decision tree model using the provided DataFrame.

    Args:
      df (DataFrame): The input DataFrame containing the training data.

    Returns:
      None
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    self.model.fit(values, labels)

  def test(self, df: DataFrame) -> float:
    """
    Test the decision tree model on the given DataFrame and return the accuracy score.

    Parameters:
    - df (DataFrame): The DataFrame containing the test data.

    Returns:
    - float: The accuracy score of the decision tree model on the test data.
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    score = self.model.score(values, labels)

    return score
  
  def predict(self, df: DataFrame) -> list:
    """
    Predicts the labels for the given DataFrame using the trained model.

    Args:
      df (DataFrame): The input DataFrame containing the feature values.

    Returns:
      list: The predicted labels for the input DataFrame.
    """
    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    return self.model.predict(values)
  
  def getDefaultParams(self) -> dict:
    """
    Get the default parameters

    :return: default parameters
    """
    return self.model.get_params()
  

