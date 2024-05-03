import os
import sys

def formatString(string: str) -> str:
  """
  Formats a string to be more readable.

  Args:
      string (str): The string to format.

  Returns:
      str: The formatted string.
  """
  metrics = [
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


def formatMetricName(string):
  metricNames = ['BCELogits', 'BCE', 'MSE', 'MAE']

  for name in metricNames:
    if name in string:
      return name
    
  return string

def getTerminalArgs(types: list = None) -> list:
  """
  Returns a list of commands that can be run from the command line.

  Returns:
      list: A list of commands that can be run from the command line.
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

def addUnits(string):
  if 'time' in string.lower():
    return f'{string} (s)'