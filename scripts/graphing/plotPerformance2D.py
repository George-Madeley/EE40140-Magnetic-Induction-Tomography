import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from utils import formatString, addUnits

import os

def plotPerformance2D(
      model: str,
      verbose: bool = False
  ) -> None:
    """
    Plots the performance of a model in 2D.

    Returns:
        None
    """

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
    features = data.columns[data.columns.get_loc('numSamples') + 1 :data.columns.get_loc('Accuracy')]

    # Get the top two features with the largest standard deviation in the
    # accuracy. This is done by creating a list of features then selecting
    # two features. The data is then grouped by these two features and the mean
    # accuracy for each group is calculated. An array of the mean accuracy is
    # then created and the standard deviation of this array is calculated. The
    # two features with the largest standard deviation are selected.
    top_features = []
    for x_feature in features:
      for y_feature in features:
        if x_feature == y_feature:
          continue
        data_grouped = data.groupby([x_feature, y_feature]).agg({'Accuracy': 'mean'}).reset_index()
        accuracy = data_grouped['Accuracy'].values
        std = np.std(accuracy)
        top_features.append((x_feature, y_feature, std))

    top_features = sorted(top_features, key=lambda x: x[2], reverse=True)
    top_features = top_features[0][:2]

    # Calculate the number of unique values for each feature
    num_unique_values = {f:0 for f in top_features}
    for feature in top_features:
        num_unique_values[feature] = len(data[feature].unique())

    # Sort the features by the number of unique values
    top_features = sorted(num_unique_values, key=num_unique_values.get)

    y_feature = top_features[0]
    x_feature = top_features[1]

    hue = 'Accuracy'
    indicators = False
    stat = 'min'
    size = 200
    cmap = 'RdYlGn'


    plt_accuracy = graphPlot(x_feature, y_feature, 'Accuracy', data, model, indicators, stat, size, cmap)
    save_dir = os.path.join('images', 'graphs', 'performance 2D')
    os.makedirs(save_dir, exist_ok=True)        
    plt_accuracy.savefig(os.path.join(save_dir, f'{formatString(model)} - Accuracy.png'))

    if verbose:
        plt_accuracy.show()

    plt.close('all')


    plt_time = graphPlot(x_feature, y_feature, 'mean_score_time', data, model, indicators, stat, size, cmap + '_r')
    plt_time.savefig(os.path.join(save_dir, f'{formatString(model)} - Time.png'))
    
    if verbose:
        plt_time.show()
    
    plt.close('all')

def graphPlot(x_feature, y_feature, hue, data, model, indicators=False, stat='mean', size=200, cmap='RdYlGn'):
  # Check if the features are in the data.
  if x_feature not in data.columns:
    print(f"Feature {x_feature} not in data.")
    return
  if y_feature not in data.columns:
    print(f"Feature {y_feature} not in data.")
    return
  if hue not in data.columns:
    print(f"Feature {hue} not in data.")
    return
  
  # find the min, max and mean of the accuracy for each combination of x and y
  data = data.groupby([x_feature, y_feature]).agg({hue: ['min', 'max', 'mean']}).reset_index()
  data.columns = [x_feature, y_feature, 'min', 'max', 'mean']

  # Set the data type of the features to string
  data[x_feature] = data[x_feature].astype(str)
  data[y_feature] = data[y_feature].astype(str)

  data = removeFeatureValues(model, data)

  # create a dummy scatter plot to define a mappable for the colorbar creation
  dummy_plot = plt.scatter([], [], c=[], cmap=cmap)

  # plot a scatter plot with the mean accuracy
  sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=stat, palette=cmap, s=size)
  plt.xlabel(formatString(x_feature))
  plt.ylabel(formatString(y_feature))
  plt.gca().set_xticklabels([formatString(label.get_text()) for label in plt.gca().get_xticklabels()])
  plt.gca().set_yticklabels([formatString(label.get_text()) for label in plt.gca().get_yticklabels()])
  plt.title(f'{formatString(model)} {formatString(hue)}')
  plt.legend().remove()
  
  if indicators:
    # Label each point with its value
    for i in range(len(data)):
      plt.text(data[x_feature][i], data[y_feature][i], round(data[stat][i], 2), ha='center', va='bottom')
  
  # Add a color bar
  colorbar = plt.colorbar(dummy_plot, label=formatString(hue))
  colorbar.set_label(addUnits(formatString(hue)))
  plt.clim(data[stat].min(), data[stat].max())

  plt.tight_layout()

  return plt

def removeFeatureValues(
      model: str,
      data: pd.DataFrame
  ):
    """
    Removes the specified values from the specified feature in the data.

    Returns:
        pd.DataFrame
    """
    if model == 'DecisionTree':
      # set 'max_depth' to int
      data['max_depth'] = data['max_depth'].astype(int)
      # remove records where 'max_depth' is 1
      data = data[data['max_depth'] != 1]

    elif model == 'RandomForest':
      # set 'max_depth' to int
      data['max_depth'] = data['max_depth'].astype(int)
       # remove records where 'max_depth' is 1
      data = data[data['max_depth'] != 1]

    elif model == 'StochasticGradientDescent':
       # remove records where 'Loss' is 'squared_hinge' or 'log'
      data = data[data['loss'] != 'squared_hinge']
      data = data[data['loss'] != 'log']

    elif model == 'SupportVectorMachine':
       # remove records where kernel is 'sigmoid'
      data = data[data['kernel'] != 'sigmoid']

    return data

if __name__ == '__main__':
    models = [
      'DecisionTree',
      'KNearestNeighbors',
      'RandomForest',
      'StochasticGradientDescent',
      'SupportVectorMachine'
    ]
    for model in models:
      plotPerformance2D(model)
    