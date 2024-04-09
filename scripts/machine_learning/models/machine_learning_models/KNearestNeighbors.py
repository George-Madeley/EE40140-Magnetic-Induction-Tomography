from sklearn.neighbors import KNeighborsClassifier
from scripts.image_generation.models.IModel import IModel
from scripts.image_generation.models.machine_learning_models.MachineLearningModel import MachineLearningModel
from pandas import DataFrame
from typing import List, Union

class KNearestNeighbor(IModel, MachineLearningModel):
  def __init__(self, k: int, weights: str = 'distance'):
    self.model = self.createModel(k, weights)

  def createModel(self, k: int, weights: str = 'distance') -> KNeighborsClassifier:
    """
    Create a k nearest neighbor model

    :param k: number of neighbors
    :param weights: weight function used in prediction

    :return: model
    """
    model = KNeighborsClassifier(n_neighbors=k, weights=weights)
    return model

  def train(self, df_train: DataFrame, noise: bool = False) -> None:
    """
    Train the model

    :param train_df: training dataframe
    :param noise: whether to include background noise
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_train)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_train)

    labels = super().getLabels(df_train)

    self.model.fit(values, labels)

  def test(self, df_test: DataFrame, noise: bool = False) -> float:
    """
    Test the model

    :param test_df: test dataframe
    :param noise: whether to include background noise

    :return: score
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_test)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_test)

    labels = super().getLabels(df_test)

    score = self.model.score(values, labels)

    return score

  def predict(self, df_predictions: DataFrame, noise: bool = False) -> List[Union[float, List[float]]]:
    """
    Predict the labels of the test data

    :param test_df: test dataframe
    :param noise: whether to include background noise

    :return: predictions
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_predictions)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_predictions)

    predictions = self.model.predict_proba(values)

    return predictions
