import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import formatString

import os
import warnings


def plotPerformance1D(
    model: str,
    verbose: bool = False,
  ) -> None:
  # Get a list of all the files in the directory 'results'
  # directory that starts with the model name.
  directory = os.path.join('results', 'classification')
  files = os.listdir(directory)
  files = [f for f in files if f.startswith(model)]
  dfs = [pd.read_csv(os.path.join(directory, f)) for f in files]

  if len(dfs) == 0:
    print(f'No data found for {model}')
    return

  data = pd.concat(dfs)

  # get the column names between 'num_samples' and 'Accuracy'
  features = data.columns[
    data.columns.get_loc('numSamples') + 1:data.columns.get_loc('Accuracy')
  ]

  graph_directory = os.path.join('images', 'graphs', 'performance 1D')
  os.makedirs(graph_directory, exist_ok=True)

  plotGraph(
      model,
      data,
      features,
      graph_directory,
      metric='mean_score_time',
      units=' (s)',
      verbose=verbose
  )
  # Get the 5% and 95% quantiles of the accuracy
  quantiles = data['Accuracy'].quantile([0.1, 0.95])
  data = data[(data['Accuracy'] > quantiles[0.1]) &
              (data['Accuracy'] < quantiles[0.95])]
  plotGraph(model, data, features, graph_directory, metric='Accuracy')


def plotGraph(
  model: str,
  data: pd.DataFrame,
  features: list,
  graphDirectory: str,
  metric: str,
  units: str = '',
  font_size: int = 16,
  verbose: bool = False,
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


    if verbose:
      print(f'({letter}) - {formatString(model)} {formatString(metric)} by {formatString(feature)}')
      print(summary)
      print('\n')


    warnings.filterwarnings("ignore")

    plt.figure(figsize=(10, 6))
    sns.boxplot(
      x=feature,
      y=metric,
      data=data,
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
