import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
import warnings


def plotPerformance1D(model: str) -> None:
  # Get a list of all the files in the directory 'results'
  # directory that starts with the model name.
  files = os.listdir('results')
  files = [f for f in files if f.startswith(model)]

  # Create a list of all the dataframes.
  dfs = [pd.read_csv(os.path.join('results', f)) for f in files]

  # Concatenate all the dataframes.
  data = pd.concat(dfs)

  # get the column names between 'num_samples' and 'Accuracy'
  features = data.columns[
    data.columns.get_loc('numSamples') + 1:data.columns.get_loc('Accuracy')
  ]

  graphDirectory = os.path.join('images', 'graphs')

  plotGraph(
      model,
      data,
      features,
      graphDirectory,
      metric='mean_score_time',
      cmap='plasma',
      units=' (s)')
  # Get the 5% and 95% quantiles of the accuracy
  quantiles = data['Accuracy'].quantile([0.1, 0.95])
  data = data[(data['Accuracy'] > quantiles[0.1]) &
              (data['Accuracy'] < quantiles[0.95])]
  plotGraph(model, data, features, graphDirectory, metric='Accuracy')


def plotGraph(
  model,
  data,
  features,
  graphDirectory,
  metric,
  cmap='plasma',
  units='',
  font_size=16
):
  for i, feature in enumerate(features):

    # Get the letter of the alphabet that corresponds to the feature index
    letter = chr(97 + i)

    if feature == 'max_depth':
      # Remove the data where the max_depth is 1
      data = data[data['max_depth'] != 1]

    if model == 'StochasticGradientDescent' and feature == 'loss':
      # Remove the data where the loss is 'log'
      data = data[data['loss'] != 'log']

    # Create a dataframe that contains the min, LQ, median, UQ, max, std, and
    # mean of the metric for each unqiue value of the features
    summary = data.groupby(feature).agg({
      metric: [
        'min',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'max',
        'std',
        'mean',
        lambda x: x.quantile(0.75) - x.quantile(0.25)
      ]
    }).reset_index()

    # print the summary
    print(f'({letter}) - {formatString(model)} {formatString(metric)} by {formatString(feature)}')
    print(summary)
    print('\n')


    warnings.filterwarnings("ignore")

    plt.figure(figsize=(10, 6))
    sns.boxplot(
      x=feature,
      y=metric,
      # hue='material',
      data=data,
      palette=cmap,
    )
    plt.xlabel(
      formatString(feature),
      fontsize=font_size + 2
    )
    plt.gca().set_xticklabels(
      [formatString(label.get_text()) for label in plt.gca().get_xticklabels()],
      fontsize=font_size
    )
    plt.ylabel(
      formatString(metric) + units,
      fontsize=font_size + 2
    )
    plt.yticks(fontsize=font_size)
    plt.title(
      f'({letter}) - {formatString(model)} {formatString(metric)} by {formatString(feature)}',
      fontsize=font_size + 4
    )
    plt.tight_layout()

    modelName = formatString(model)
    metricName = formatString(metric)
    featureName = formatString(feature)

    filepath = os.path.join(graphDirectory, f'{modelName} - {metricName} - {featureName}.png')
    plt.savefig(filepath)
    # plt.show()
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
  for i in range(1, len(string)):
    if string[i].isupper() and string[i - 1] != ' ':
      string = string[:i] + ' ' + string[i:]
  return string.replace('_', ' ').title()


if __name__ == '__main__':
  models = [
    'DecisionTree',
    'KNearestNeighbors',
    'NearestCentroid',
    'NeuralNetwork',
    'RandomForest',
    'StochasticGradientDescent',
    'SupportVectorMachine'
  ]
  for model in models:
    print("-----------------------------------")
    print(f'Plotting performance for {model}')
    plotPerformance1D(model)
