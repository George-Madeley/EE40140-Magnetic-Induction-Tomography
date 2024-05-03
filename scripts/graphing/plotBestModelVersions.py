from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import os

def plotBestModelVersions():
  models = {
    'NN': 'ANN',
    'GAN': 'G',
    'VAE': 'VAE'
  }

  directory = os.path.join('results', 'generation')
  files = os.listdir(directory)

  for model, id in models.items():
    model_files = [file for file in files if file.startswith(model)]

    df = pd.DataFrame()

    # metric
    metric = f'test {id} MAE Loss'

    for model_file in model_files:
      model_df = pd.read_csv(os.path.join(directory, model_file))

      modelId = model_file.split(' - ')[0]

      model_df['Model Name'] = modelId

      # Find the records in each group with the min metric
      model_df = model_df[model_df[metric] == model_df[metric].min()]
      model_df = model_df.reset_index(drop=True)
      df = pd.concat([df, model_df])

    # remove all the columns with 'train' in the name
    df = df[df.columns.drop(list(df.filter(regex='train')))]

    # replace all the columns with 'test' in the name with ''
    df.columns = df.columns.str.replace('test ', '')
    df.columns = df.columns.str.replace(f'{id} ', '')

    # format all the column names
    df.columns = [formatString(col) for col in df.columns]

    # save the dataframe to a csv file
    savepath = os.path.join('results', 'generation', f'{model} - best.csv')
    df.to_csv(savepath, index=False)

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

if __name__ == '__main__':
  plotBestModelVersions()