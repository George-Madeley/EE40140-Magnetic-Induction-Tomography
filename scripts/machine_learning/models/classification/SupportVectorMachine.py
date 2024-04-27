from typing import List, Literal, Union

from sklearn.svm import SVC
from pandas import DataFrame

from .IClassification import IClassification

class SupportVectorMachine(IClassification):
  def __init__(
      self,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = False,
      oneHotEncode: bool = False
    ):
    """
    Initializes a SupportVectorMachine object.

    Returns:
    - None
    """
    self.validParams = {
      'C': [0.1, 1, 10, 100],
      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
      'degree': list(range(1, 6)),
      'cache_size': [200, 400, 600, 800, 1000],
    }
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.model = SVC()
  
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
    Predict the labels of the test data

    :param test_df: test dataframe
    :param noise: whether to include background noise
    """
    values, _ = super().getValuesAndLabels(df, self.labelName, self.noise)

    return self.model.predict(values)
  
  def getDefaultParams(self):
    """
    Get the default parameters for the model
    
    Returns:
    - dict: default parameters
    """
    return self.model.get_params()