from typing import Literal
from sklearn.ensemble import RandomForestClassifier
from IModel import IModel

class RandomForest(IModel):
  def __init__(
    self,
    max_depth: int = None,
    splitter: Literal['best', 'random'] = 'best',
    criterion: Literal['gini', 'entropy'] = 'gini',
    min_samples_split: float | int = 2,
    min_samples_leaf: float | int = 1,
    max_features: float | int | str = None,
    n_estimators: int = 100,
  ):
    """
    Initialize the RandomForest model with the specified parameters.

    Parameters:
    - max_depth (int): The maximum depth of the tree. If None, the tree is fully grown.
    - splitter (str): The strategy used to choose the split at each node. Can be 'best' or 'random'.
    - criterion (str): The function to measure the quality of a split. Can be 'gini' or 'entropy'.
    - min_samples_split (float or int): The minimum number of samples required to split an internal node.
    - min_samples_leaf (float or int): The minimum number of samples required to be at a leaf node.
    - max_features (float, int, or str): The number of features to consider when looking for the best split.
    - n_estimators (int): The number of trees in the forest.

    Returns:
    None
    """
    super().enforceLiterals(self.__init__)
    super().__init__()

    self.randomForest = RandomForestClassifier(
      max_depth=max_depth,
      splitter=splitter,
      criterion=criterion,
      min_samples_split=min_samples_split,
      min_samples_leaf=min_samples_leaf,
      max_features=max_features,
      n_estimators=n_estimators
    )

  def train(self, df_train, noise=False):
    """
    Trains the random forest model using the provided training data.

    Args:
      df_train (DataFrame): The training data as a pandas DataFrame.
      noise (bool, optional): Flag indicating whether to include background noise in the training data. 
                  Defaults to False.

    Returns:
      None
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_train)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_train)

    labels = super().getLabels(df_train)

    self.randomForest.fit(values, labels)

  def test(self, df_test, noise=False):
    """
    Test the random forest model on the given test dataset.

    Parameters:
    - df_test (DataFrame): The test dataset.
    - noise (bool): Flag indicating whether to add background noise to the test dataset.

    Returns:
    - score (float): The accuracy score of the random forest model on the test dataset.
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_test)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_test)

    labels = super().getLabels(df_test)

    score = self.randomForest.score(values, labels)

    return score
  
  def predict(self, df_predict, noise=False):
    """
    Predicts the target variable for the given input data.

    Args:
      df_predict (pandas.DataFrame): The input data to make predictions on.
      noise (bool, optional): Whether to add background noise to the input data. 
                  Defaults to False.

    Returns:
      numpy.ndarray: The predicted target variable values.
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_predict)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_predict)

    return self.randomForest.predict(values)
  
  @staticmethod
  def isParamValid(self, name: str, value: str) -> bool:
    """
    Check if the parameter is valid

    :param name: name of the parameter
    :param value: value of the parameter

    :return: boolean
    """
    validParams = {
      'n_estimators': list(range(1, 1001)),
      'criterion': ['gini', 'entropy', 'log_loss'],
      'max_depth': list(range(1, 101)),
      'min_samples_split': list(range(2, 21)),
      'min_samples_leaf': list(range(1, 21)),
      'max_features': ['sqrt', 'log2', None],
      'max_leaf_nodes': list(range(2, 101)),
      'min_impurity_decrease': list(range(0, 1, 0.01)),
    }

    if name in validParams.keys() and value in validParams.get(name, []):
      return True
    return False
  
  def getDefaultParams(self) -> dict:
    """
    Get the default parameters

    :return: default parameters
    """
    return self.model.get_params()
    