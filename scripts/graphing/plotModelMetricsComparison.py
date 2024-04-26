import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

def plotModelMetricsComparison(font_size=16) -> None:
  models = {
    'DecisionTree': {
      'criterion': 'entropy',
      'max_depth': 201,
      'splitter': 'best'
    },
    'KNearestNeighbors': {
      'algorithm': 'brute',
      'metric': 'manhattan',
      'n_neighbors': 3,
      'weights': 'distance'
    },
    'NearestCentroid': {
      'metric': 'chebyshev',
    },
    'NeuralNetwork': {
      'activation': 'relu',
      'alpha': 0.0001,
      'hidden_layer_sizes': '(100, 100)',
      'max_iter': 600,
      'learning_rate': 'constant',
    },
    'RandomForest': {
      'criterion': 'log_loss',
      'max_depth': 601,
      'n_estimators': 101,
    },
    'StochasticGradientDescent': {
      'epsilon': 0.31,
      'loss': 'perceptron',
      'max_iter': 600,
      'penalty': 'l2',
    },
    'SupportVectorMachine': {
      'cache_size': 800,
      'degree': 5,
      'kernel': 'sigmoid',
    }
  }
  
  model_data = []

  # for each model in models.keys(), get the results files from 'results' directory and
  # concatenate them into a single dataframe. The group the dataframe by the parameters
  # defined in the model dictionary.
  for model, params in models.items():
    files = os.listdir('results')
    files = [f for f in files if f.startswith(model)]
    dfs = [pd.read_csv(os.path.join('results', f)) for f in files]
    data = pd.concat(dfs)

    if model == 'NeuralNetwork':
      # set the hidden_layer_sizes to a string
      data['hidden_layer_sizes'] = data['hidden_layer_sizes'].astype(str)

    # check if each key in the params dictionary is in the columns of the dataframe
    for key, value in params.items():
      if key not in data.columns:
        raise ValueError(f'{key} not in columns of dataframe')
      else:
        # check if the value of the key is in the unique values of the column
        if value not in data[key].unique():
          raise ValueError(f'{value} not in unique values of {key}')

    keepColumns = ['model', 'material', 'numSamples', 'Accuracy', 'F1', 'Precision', 'Recall', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']
          
    # Get the record that matches the parameters in the model dictionary
    data = data[
      (data[list(params.keys())] == pd.Series(params)).all(axis=1)
    ]

    # drop the columns that are not in keepColumns
    data = data[keepColumns]
    model_data.append(data)

  # concatenate all the dataframes in model_data
  data = pd.concat(model_data).reset_index(drop=True)

  # # drop the material and numSamples columns
  # data = data.drop(columns=['material', 'numSamples'])

  # # group the data by the model and get the mean of each group
  # data = data.groupby('model').mean().reset_index()

  # # save the data to a csv file
  # data.to_csv('model_metrics_comparison.csv', index=False)

  metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'mean_fit_time', 'mean_score_time']

  # create a barplot for each metric in metrics where the x-axis is the model and the y-axis is the metric
  for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y=metric, data=data, palette='plasma')
    plt.xlabel('Model', fontsize = font_size + 2)
    plt.ylabel(formatString(metric), fontsize = font_size + 2)
    plt.gca().set_xticklabels(
      [formatString(label.get_text()) for label in plt.gca().get_xticklabels()],
      fontsize=font_size,
      rotation=45
    )
    plt.yticks(fontsize=font_size)
    plt.title(f'{metric} Comparison for Classification Models', fontsize=font_size + 4)
    filepath = os.path.join('images', 'graphs', f'Comparison - {metric} - Barchart.png')
    plt.savefig(filepath)
    plt.close()

  # create a boxplot for each metric in metrics where the y-axis is the metric
  # and the x-axis is each combination of material and numSamples
  for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='material', y=metric, hue='numSamples', data=data)
    plt.xlabel('Material', fontsize=font_size + 2)
    plt.ylabel(formatString(metric), fontsize=font_size + 2)
    plt.gca().set_xticklabels(
      [formatString(label.get_text()) for label in plt.gca().get_xticklabels()],
      fontsize=font_size,
    )
    plt.yticks(fontsize=font_size)
    plt.title(f'{metric} Comparison for Classification Models', fontsize=font_size + 4)
    filepath = os.path.join('images', 'graphs', f'Comparison - {metric} - Boxplot.png')
    plt.savefig(filepath)
    plt.close()

def formatString(string: str) -> str:
  """
  Formats a string to be more readable.

  Args:
      string (str): The string to format.

  Returns:
      str: The formatted string.
  """
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
  plotModelMetricsComparison()