import os
import sys
from typing import List

def formatString(string: str) -> str:
  """
  Formats a string to be more readable.

  Args:
    string (str): The string to format.

  Returns:
    str: The formatted string.
  """
  metrics: List[str] = [
    'MAE Loss',
    'MSE Loss',
    'BCE Loss',
    'BCELogits Loss'
  ]
  if string in metrics:
    return string

  if string == 'iron':
    string = 'Aluminum'

  # if there is a captial letter in the string that is not the first letter
  # add a space before it
  modelInitials = {
    'DecisionTree': 'DT',
    'KNearestNeighbors': 'KNN',
    'NearestCentroid': 'NC',
    'NeuralNetwork': 'NN',
    'RandomForest': 'RF',
    'StochasticGradientDescent': 'SGD',
    'SupportVectorMachine': 'SVM'
  }
  if string in modelInitials.keys():
    return modelInitials[string]

  for i in range(1, len(string)):
    if string[i].isupper() and string[i - 1] != ' ':
      string = string[:i] + ' ' + string[i:]
  return string.replace('_', ' ').title()


def formatMetricName(string: str) -> str:
  """
  Formats a metric name to be more readable.

  Args:
    string (str): The metric name to format.

  Returns:
    str: The formatted metric name.
  """
  metricNames = ['BCELogits', 'BCE', 'MSE', 'MAE', 'SSIM']

  for name in metricNames:
    if name in string:
      return name
    
  return string

def getTerminalArgs(types: List[str] = None) -> List[str]:
  """
  Retrieves the command line arguments passed to the script.

  Args:
    types (List[str], optional): A list of types to cast the arguments to. Defaults to None.

  Returns:
    List[str]: The command line arguments.
  """
  args = sys.argv[2:]

  if types:
    for i in range(len(args)):
      if types[i] == 'int':
        args[i] = int(args[i])
      elif types[i] == 'float':
        args[i] = float(args[i])
      elif types[i] == 'bool':
        args[i] = args[i] == 'True'
      else:
        args[i] = str(args[i])

  return args

def addUnits(string: str) -> str:
  """
  Adds units to a string if it contains the word 'time'.

  Args:
    string (str): The string to add units to.

  Returns:
    str: The string with units added.
  """
  if 'time' in string.lower():
    return f'{string} (s)'
