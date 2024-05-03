from abc import ABC, abstractmethod
from pandas import DataFrame

class IModel(ABC):
  def __init__(self) -> None:
    self.labelNames = []

  @abstractmethod
  def train(self, df: DataFrame) -> None:
    """
    Train the model

    :param train_df: training dataframe
    """
    pass

  @abstractmethod
  def test(self, df: DataFrame) -> None:
    """
    Test the model

    :param test_df: test dataframe

    :return: metrics
    """
    pass

  @abstractmethod
  def predict(self, df: DataFrame) -> None:
    """
    Predict the labels of the test data

    :param predict_df: prediction dataframe

    :return: predictions
    """
    pass

  def getValuesAndLabels(
    self,
    df: DataFrame,
  ):
    """
    Get the values and labels from the dataframe

    :param df: dataframe

    :return: values, labels
    """

    valueColumnNames = df.filter(regex='cc_\d+').columns
    if self.noise:
      valueColumnNames = valueColumnNames.append(df.filter(regex='bb_\d+').columns)

    values = df[valueColumnNames].values

    if self.oneHotEncode:
      labelColumnNames = df.filter(regex=f'^{self.labelName}_').columns
      self.labelNames = [label[-1] for label in labelColumnNames]
      labels = df[labelColumnNames].values
    else:
      labels = df[self.labelName].values

    return values, labels