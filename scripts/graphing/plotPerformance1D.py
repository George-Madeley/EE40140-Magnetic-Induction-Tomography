import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

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
    features = data.columns[data.columns.get_loc('numSamples') + 1 :data.columns.get_loc('Accuracy')]

    graphDirectory = os.path.join('images', 'graphs')

    plotGraph(model, data, features, graphDirectory, metric='mean_score_time', cmap='plasma', units=' (s)')
    # Get the 5% and 95% quantiles of the accuracy
    quantiles = data['Accuracy'].quantile([0.1, 0.95])
    data = data[(data['Accuracy'] > quantiles[0.1]) & (data['Accuracy'] < quantiles[0.95])]
    plotGraph(model, data, features, graphDirectory, metric='Accuracy')

def plotGraph(model, data, features, graphDirectory, metric, cmap='plasma', units=''):
    for feature in features:

      if feature == 'max_depth':
        # Remove the data where the max_depth is 1
        data = data[data['max_depth'] != 1]

      if model == 'StochasticGradientDescent' and feature == 'loss':
        # Remove the data where the loss is 'log'
        data = data[data['loss'] != 'log']

      plt.figure(figsize=(10, 6))
      sns.boxplot(x=feature, y=metric, data=data, palette=cmap)
      plt.xlabel(formatString(feature))
      plt.gca().set_xticklabels([formatString(label.get_text()) for label in plt.gca().get_xticklabels()])
      plt.ylabel(formatString(metric) + units)
      plt.title(f'{formatString(model)} {formatString(metric)} by {formatString(feature)}')
      plt.tight_layout()

      filepath = os.path.join(graphDirectory, f'{model}_{feature} - {metric}.png')
      plt.savefig(filepath)
      plt.show()


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
      if string[i].isupper() and string[i-1] != ' ':
        string = string[:i] + ' ' + string[i:]
    return string.replace('_', ' ').title()

if __name__ == '__main__':
  model = 'SupportVectorMachine'
  plotPerformance1D(model)