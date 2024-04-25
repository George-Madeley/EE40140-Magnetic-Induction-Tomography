import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import os

def plotPerformance2D(model: str) -> None:
    """
    Plots the performance of a model in 2D.

    Returns:
        None
    """

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

    # cmap options: 'plasma', 'binary', 'YlGn'

    print('\n\n Filename: ')
    fileName = str(input('>?\t'))

    plt_accuracy = graphPlot(x_feature, y_feature, 'Accuracy', data, model, indicators, stat, size, cmap)
    if fileName != '':
        filePath = os.path.join('images', 'graphs', f'{fileName} - Accuracy.png')
        plt_accuracy.savefig(filePath)
    else:
        plt_accuracy.show()

    plt.close('all')


    plt_time = graphPlot(x_feature, y_feature, 'mean_score_time', data, model, indicators, stat, size, cmap + '_r')
    if fileName != '':
        filePath = os.path.join('images', 'graphs', f'{fileName} - Time.png')
        plt_time.savefig(filePath)
    else:
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

  # remove rows where max_depth is 1
  # data = data[data[x_feature] != '1'].reset_index(drop=True)
  # data = data[data[y_feature] != 'log'].reset_index(drop=True)


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
  plt.colorbar(dummy_plot, label=formatString(hue))
  plt.clim(data[stat].min(), data[stat].max())

  plt.tight_layout()

  return plt

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
    model = 'DecisionTree'
    plotPerformance2D(model)
    