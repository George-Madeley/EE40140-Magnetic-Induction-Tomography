from typing import Literal
from sklearn.tree import DecisionTreeClassifier
from MachineLearningModel import MachineLearningModel
from IMLModel import IModel

class DecisionTrees(IModel, MachineLearningModel):
  def __init__(
    self,
    max_depth: int = None,
    splitter: Literal['best', 'random'] = 'best',
    criterion: Literal['gini', 'entropy'] = 'gini',
    min_samples_split: float | int = 2,
    min_samples_leaf: float | int = 1,
    max_features: float | int | str = None,
  ):
    """
    Initialize a DecisionTrees object.

    Args:
      max_depth (int, optional): The maximum depth of the decision tree. Defaults to None.
      splitter (Literal['best', 'random'], optional): The strategy used to choose the split at each node. Defaults to 'best'.
      criterion (Literal['gini', 'entropy'], optional): The function to measure the quality of a split. Defaults to 'gini'.
      min_samples_split (float | int, optional): The minimum number of samples required to split an internal node. Defaults to 2.
      min_samples_leaf (float | int, optional): The minimum number of samples required to be at a leaf node. Defaults to 1.
      max_features (float | int | str, optional): The number of features to consider when looking for the best split. Defaults to None.
    """
    super().enforceLiterals(self.__init__)
    super().__init__()

    self.dtree = DecisionTreeClassifier(
      max_depth=max_depth,
      splitter=splitter,
      criterion=criterion,
      min_samples_split=min_samples_split,
      min_samples_leaf=min_samples_leaf,
      max_features=max_features
    )

  def train(self, df_train, noise=False):
    """
    Trains the decision tree model using the provided training data.

    Args:
      df_train (DataFrame): The training data as a pandas DataFrame.
      noise (bool, optional): Flag indicating whether to include background noise in the training data.

    Returns:
      None
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_train)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_train)

    labels = super().getLabels(df_train)

    self.dtree.fit(values, labels)

  def test(self, df_test, noise=False):
    """
    Test the decision tree model on a given test dataset.

    Parameters:
    - df_test (pandas.DataFrame): The test dataset to evaluate the model on.
    - noise (bool): Flag indicating whether to add background noise to the test dataset.

    Returns:
    - score (float): The accuracy score of the model on the test dataset.
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_test)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_test)

    labels = super().getLabels(df_test)

    score = self.dtree.score(values, labels)

    return score
  
  def predict(self, df_predictions, noise=False):
    """
    Predicts the target variable for the given input data.

    Args:
      df_predictions (pandas.DataFrame): The input data for making predictions.
      noise (bool, optional): Flag indicating whether to add background noise to the input data. 
                  Defaults to False.

    Returns:
      numpy.ndarray: The predicted target variable values.
    """
    if noise:
      values = super().getValuesWithBackgroundNoise(df_predictions)
    else:
      values = super().getValuesWithoutBackgroundNoise(df_predictions)

    return self.dtree.predict(values)
  
  def printModelParams(self):
    """
    Prints the parameters of the decision tree model.

    Returns:
      None
    """
    params = self.dtree.get_params()
    print(params)

  def getDepth(self):
    """
    Returns the depth of the decision tree.

    Returns:
      int: The depth of the decision tree.
    """
    return self.dtree.get_depth()
  
  def plotTree(self):
    """
    Plots the decision tree model.

    This method uses the `plot_tree` function from the `sklearn.tree` module to visualize the decision tree model.
    The resulting plot is displayed using `matplotlib.pyplot`.

    Parameters:
    None

    Returns:
    None
    """
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(self.dtree, ax=ax)
    plt.show()