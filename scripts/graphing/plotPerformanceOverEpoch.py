from typing import Literal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import formatString, formatMetricName

import os

def plotPerformanceOverEpoch(
    model: Literal['NN', 'GAN', 'VAE'] = 'NN',
    metric: Literal['BCE', 'BCELogits', 'MSE', 'MAE'] = 'BCE',
    verbose: bool = False
):
  directory = os.path.join('results', 'generation')
  files = os.listdir(directory)
  files = [f for f in files if f.startswith(model)]

  df = pd.DataFrame()
  for f in files:
    data = pd.read_csv(os.path.join(directory, f))

    # find the model number
    modelName = data['Model Name'].iloc[0]
    modelID = f.split(' - ')[0]
    modelNum = int(modelID.replace(model, ''))
    data['modelNum'] = modelNum

    # Each dataframe has metrics for training and testing.
    # We want to modify the dataframe to have a column for each metric and a
    # column for the mode (training or testing). Therefore, all of the testing
    # metrics will be in the same column as the training metrics.
    # first, we get the columns from the start to, and including, 'epoch'
    columns = data.columns.tolist()
    epoch_index = columns.index('Epoch')
    metaColumns = columns[:epoch_index + 1]
    trainColumns = [c for c in columns if 'train' in c]
    testColumns = [c for c in columns if 'test' in c]

    dfs = []

    if len(trainColumns) // 4 > 1:
      subNetworks = []
      for colName in trainColumns:
        subNetwork = colName.split(' ')[1]
        if subNetwork not in subNetworks:
          subNetworks.append(subNetwork)
      
      for subNetwork in subNetworks:
        subNetworkColumns = [c for c in columns if f'train {subNetwork}' in c]
        # Create a new dataframe with the data from data in the columns stated in
        # trainColumns and metaColumns
        subNetworkData = data[metaColumns + subNetworkColumns]
        subNetworkData['mode'] = f'{subNetwork} train'
        subNetworkData = subNetworkData.rename(columns={c: formatMetricName(c) for c in subNetworkColumns})
        dfs.append(subNetworkData)
    else:
      # Create a new dataframe with the data from data in the columns stated in
      # trainColumns and metaColumns
      trainData = data[metaColumns + trainColumns]
      trainData['mode'] = 'train'
      trainData = trainData.rename(columns={c: formatMetricName(c) for c in trainColumns})
      dfs.append(trainData)

        

    if len(testColumns) // 4 > 1:
      subNetworks = []
      for colName in testColumns:
        subNetwork = colName.split(' ')[1]
        if subNetwork not in subNetworks:
          subNetworks.append(subNetwork)
      
      for subNetwork in subNetworks:
        subNetworkColumns = [c for c in columns if f'test {subNetwork}' in c]
        # Create a new dataframe with the data from data in the columns stated in
        # trainColumns and metaColumns
        subNetworkData = data[metaColumns + subNetworkColumns]
        subNetworkData['mode'] = f'{subNetwork} test'
        subNetworkData = subNetworkData.rename(columns={c: formatMetricName(c) for c in subNetworkColumns})
        dfs.append(subNetworkData)
    else:
      # Create a new dataframe with the data from data in the columns stated in
      # testColumns and metaColumns
      testData = data[metaColumns + testColumns]
      testData['mode'] = 'test'
      testData = testData.rename(columns={c: formatMetricName(c) for c in testColumns})
      dfs.append(testData)

  
    # Concatenate the two dataframes
    data = pd.concat(dfs)

    # plot the data to a line plot where the x-axis is the epoch and the y-axis
    # is the metric value and the hue is the mode
    sns.lineplot(data=data, x='Epoch', y=metric, hue='mode')
    plt.title(f'{formatString(modelName)} - {modelID}')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    saveDir = os.path.join('images', 'graphs', 'performance', model, metric)
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(os.path.join(saveDir, f'{modelID} - {metric}.png'))
    
    if verbose:
      plt.show()

    plt.close()

if __name__ == '__main__':
  models = ['NN', 'GAN', 'VAE']
  metrics = ['BCE', 'BCELogits', 'MSE', 'MAE']

  for model in models:
    for metric in metrics:
      plotPerformanceOverEpoch(model=model, metric=metric)