from typing import Union
import time
import csv
import os

import pandas as pd

from models.machine_learning_models.IMLModel import IModel
from models.machine_learning_models.MachineLearningModel import MachineLearningModel

def runModel(
  modelClass: type[Union[IModel, MachineLearningModel]],
  df: pd.DataFrame,
  noise: bool = False, 
  **kwargs
):
  """
  Runs the model with different parameter values and logs the results.

  Args:
    modelClass (type[Union[IModel, MachineLearningModel]]): The class of the model to be tested.
    df (pd.DataFrame): The input data for training and evaluation.
    noise (bool, optional): Flag indicating whether to add noise to the data. Defaults to False.
    **kwargs: Additional keyword arguments representing the parameters and their corresponding values.

  Returns:
    None
  """
  # Identify the valid parameters for the model i.e., the parameters that the
  # model can accept and can vary. The parameters that the model cannot accept
  # or cannot vary are added to the params dictionary with a value of 'n/a'.
  keys = kwargs.keys()
  params = {}
  for key in keys:
    if not modelClass.isParamValid(key, kwargs[key]):
      params[key] = 'n/a'
    else:
      params[key] = kwargs[key]

  # Filter out the parameters that are not valid
  validParams = {param:args for param, args in params.items() if args is not 'n/a'}

  # Loop through the valid parameters to test the model
  # with different parameter values
  for param, args in validParams.items():
    
    # Loop through the arguments for the parameter
    for arg in args:

      # Start the timer
      startTime = time.perf_counter_ns()

      # Create the model
      model = modelClass(param=arg)
      model.train(df, noise=noise)
      predictions, actual = model.predict(df, noise=noise)
      metrics = model.evaluate(df, noise=noise)

      # End the timer
      endTime = time.perf_counter_ns()

      # Log the results
      defaultParams = model.getDefaultParams()
      defaultParams[param] = arg
      logResults(modelClass.__name__, endTime - startTime, metrics, defaultParams)

def logResults(modelName: str, runtime: float | int, metrics: dict, variables: dict):
  """
  Logs the results of a machine learning model to a CSV file.

  Args:
    modelName (str): The name of the model.
    runtime (float | int): The runtime of the model in seconds.
    metrics (dict): A dictionary containing the metrics of the model.
    variables (dict): A dictionary containing the variables used in the model.

  Returns:
    None
  """
  headers = ['Model', 'Runtime'] + list(metrics.keys()) + list(variables.keys())
  values = [modelName, runtime] + list(metrics.values()) + list(variables.values())
  filename = f'{modelName}.csv'
  filepath = os.path.join('results', filename)

  if not os.path.exists(filepath):
    with open(filepath, 'w', newline='') as f:
      csvWriter = csv.DictWriter(f, fieldnames=headers)
      csvWriter.writeheader()

  with open(filepath, 'a', newline='') as f:
    csvWriter = csv.DictWriter(f, fieldnames=headers)
    csvWriter.writerow(dict(zip(headers, values)))