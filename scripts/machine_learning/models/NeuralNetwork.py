from typing import List, Literal, Union

from sklearn.neural_network import MLPClassifier
from pandas import DataFrame

from .IModel import IModel

class NeuralNetwork(IModel):
  def __init__(
      self,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = False,
      oneHotEncode: bool = False
  ):
    """
    Initializes a NeuralNetwork object.
    
    Returns:
    - None
    """
    self.validParams = {
      'hidden_layer_sizes': [(100,), (100, 100), (100, 100, 100)],
      'activation': ['identity', 'logistic', 'tanh', 'relu'],
      'solver': ['lbfgs', 'sgd', 'adam'],
      'alpha': [0.0001, 0.001, 0.01, 0.1],
      'learning_rate': ['constant', 'invscaling', 'adaptive'],
      'max_iter': list(range(100, 1001, 100)),
      'early_stopping': [True, False],
    }
    self.labelName = labelName
    self.noise = noise
    self.oneHotEncode = oneHotEncode
    self.model = MLPClassifier(hidden_layer_sizes=(100,))

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

    :param predict_df: prediction dataframe

    :return: predictions
    """
    values, _ = super().getValuesAndLabels(df, self.labelName, False)

    predictions = self.model.predict(values)

    return predictions
  
  def getDefaultParams(self) -> dict:
    """
    Get the default parameters

    :return: default parameters
    """
    return self.model.get_params()
  