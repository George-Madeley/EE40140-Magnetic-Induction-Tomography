import matplotlib.pyplot as plt
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

    # define the features for the x and y axes
    x_feature = 'max_depth'
    y_feature = 'criterion'
    hue = 'Accuracy'
    indicators = False
    stat = 'min'
    size = 200
    cmap = 'RdBu'

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

    # remove rows where max_depth is 1
    data = data[data[x_feature] != 1].reset_index(drop=True)

    # create a dummy scatter plot to define a mappable for the colorbar creation
    dummy_plot = plt.scatter([], [], c=[], cmap=cmap)

    # plot a scatter plot with the mean accuracy
    sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=stat, palette=cmap, s=size)
    plt.xlabel(x_feature.replace('_', ' ').title())
    plt.ylabel(y_feature.replace('_', ' ').title())
    plt.title(f'{model} Performance')
    plt.legend().remove()
    
    if indicators:
      # Label each point with its value
      for i in range(len(data)):
        plt.text(data[x_feature][i], data[y_feature][i], round(data[stat][i], 2), ha='center', va='bottom')
    
    # Add a color bar
    plt.colorbar(dummy_plot, label=hue)
    plt.clim(data[stat].min(), data[stat].max())
    
    plt.show()

if __name__ == '__main__':
    model = 'DecisionTree'
    plotPerformance2D(model)
    