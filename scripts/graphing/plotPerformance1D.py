from typing import List
import seaborn as sns
import pandas as pd
from utils import formatString
import os
import warnings

import matplotlib.pyplot as plt


def plotPerformance1D(model: str, verbose: bool = False) -> None:
  """
  Plot the performance of a model in 1D.

  Args:
    model (str): The name of the model.
    verbose (bool, optional): Whether to print verbose output. Defaults to False.
  """
  directory: str = os.path.join('results', 'classification')
  files: List[str] = os.listdir(directory)
  files = [f for f in files if f.startswith(model)]
  dfs: List[pd.DataFrame] = [pd.read_csv(
    os.path.join(directory, f)) for f in files]

  if len(dfs) == 0:
    print(f'No data found for {model}')
    return

  data: pd.DataFrame = pd.concat(dfs)

  features = data.columns[
    data.columns.get_loc('numSamples') + 1:data.columns.get_loc('Accuracy')
  ]

  graph_directory: str = os.path.join('images', 'graphs', 'performance 1D')
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

  quantiles: pd.Series = data['Accuracy'].quantile([0.1, 0.95])
  data = data[(data['Accuracy'] > quantiles[0.1]) &
              (data['Accuracy'] < quantiles[0.95])]
  plotGraph(model, data, features, graph_directory, metric='Accuracy')


def plotGraph(
  model: str,
  data: pd.DataFrame,
  features: List[str],
  graphDirectory: str,
  metric: str,
  units: str = '',
  font_size: int = 16,
        verbose: bool = False) -> None:
  """
  Plot a graph for a given model and data.

  Args:
    model (str): The name of the model.
    data (pd.DataFrame): The data to plot.
    features (List[str]): The list of features.
    graphDirectory (str): The directory to save the graph.
    metric (str): The metric to plot.
    units (str, optional): The units of the metric. Defaults to ''.
    font_size (int, optional): The font size of the plot. Defaults to 16.
    verbose (bool, optional): Whether to print verbose output. Defaults to False.
  """
  for i, feature in enumerate(features):
    letter = chr(97 + i)

    if feature == 'max_depth':
      data = data[data['max_depth'] != 1]

    if model == 'StochasticGradientDescent' and feature == 'loss':
      data = data[data['loss'] != 'log']

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
      print(
        f'({letter}) - {formatString(model)} {formatString(metric)} by {formatString(feature)}')
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
        fontsize=font_size + 4)
    plt.tight_layout()

    modelName = formatString(model)
    metricName = formatString(metric)
    featureName = formatString(feature)

    filepath = os.path.join(graphDirectory,
                            f'{modelName} - {metricName} - {featureName}.png')
    plt.savefig(filepath)
    plt.close()


if __name__ == '__main__':
  models: List[str] = [
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
