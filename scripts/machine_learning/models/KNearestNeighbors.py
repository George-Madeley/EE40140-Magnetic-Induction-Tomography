from random import choice
from string import ascii_letters
from typing import List, Literal, Union

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame, concat

from .IModel import IModel
class KNearestNeighbors(IModel):
  def __init__(
      self,
      k: int,
      labelName: Literal['shape', 'sample'] = 'shape',
      noise: bool = False
    ):
    """
    Initializes a KNearestNeighbors object.

    Parameters:
    - k (int): The number of nearest neighbors to consider.

    Returns:
    - None
    """
    self.labelName = labelName
    self.noise = noise
    self.model = KNeighborsClassifier(n_neighbors=k)

  def train(self, df: DataFrame) -> None:
    """
    Train the model

    :param train_df: training dataframe
    :param noise: whether to include background noise
    """
    values, labels = super().getValuesAndLabels(df, self.labelName. self.noise)

    self.model.fit(values, labels)

  def test(self, df: DataFrame) -> float:
    """
    Test the model

    :param test_df: test dataframe
    :param noise: whether to include background noise

    :return: score
    """
    values, labels = super().getValuesAndLabels(df, self.labelName. self.noise)

    score = self.model.score(values, labels)

    return score

  def predict(self, df: DataFrame) -> List[Union[float, List[float]]]:
    """
    Predict the labels of the test data

    :param test_df: test dataframe
    :param noise: whether to include background noise

    :return: predictions
    """
    values, labels = super().getValuesAndLabels(df, self.labelName. self.noise)

    predictions = self.model.predict_proba(values)

    return predictions, labels

  def varyParams(
      self,
      df: DataFrame,
      params: dict,
      searchCV: type[GridSearchCV | RandomizedSearchCV],
      scoring: dict = None,
      n_jobs: int = -1,
      verbose: int = 0
    ) -> None:
    """
    Vary the parameters of the model

    :param params: parameters
    """
    validParams = {}
    for param in params.keys():
      if self.isParamValid(param):
        validParams[param] = params[param]

    validParams

    if validParams == {}:
      print(
        f'No valid parameters to vary for for {self.__class__.__name__} model.')

    if scoring is None:
      scoring = {
        'Precision': 'precision_micro',
        'Recall': 'recall_micro',
        'Accuracy': 'accuracy',
        'F1': 'f1_micro',
        'AUC': 'roc_auc',
      }

    values, labels = super().getValuesAndLabels(df, self.labelName, self.noise)

    clf = searchCV(
      self.model,
      validParams,
      scoring=scoring,
      n_jobs=n_jobs,
      verbose=verbose,
      refit=False
    )
    clf.fit(values, labels)


    df_results = concat(
      [
        DataFrame(clf.cv_results_['params'])
      ] + [
        DataFrame(
          clf.cv_results_[f'mean_test_{metric}'],
          columns=[metric]
        ) for metric in scoring.keys()
      ],
      axis=1
    )

    # Add the column 'model' to the dataframe and set it to the model name
    df_results['model'] = self.__class__.__name__

    # Generate a random string
    randomString = ''.join(
      [choice(ascii_letters) for i in range(10)]
    )

    # Save the results to a CSV file
    df_results.to_csv(
      f'{self.__class__.__name__}_results_{randomString}.csv',
      index=False
    )

  def isParamValid(self, name: str) -> bool:
    """
    Check if the parameter is valid

    :param name: name of the parameter

    :return: boolean
    """
    validParams = [
      'weights',
      'algorithm',
      'leaf_size'
      'p',
      'metric',
      'n_jobs',
      'n_neighbors'
    ]

    if name in validParams:
      return True
    return False

  def getDefaultParams(self) -> dict:
    """
    Get the default parameters

    :return: default parameters
    """
    return self.model.get_params()
