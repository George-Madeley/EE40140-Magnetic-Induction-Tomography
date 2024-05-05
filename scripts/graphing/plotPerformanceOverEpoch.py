from typing import Literal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import formatString, formatMetricName

import os

def plotPerformanceOverEpoch(
    material: Literal['copper', 'aluminium'] = 'copper',
    model: Literal['NN', 'GAN', 'VAE'] = 'NN',
    metric: Literal['BCE', 'BCELogits', 'MSE', 'MAE'] = 'BCE',
    verbose: bool = False
):
  directory = os.path.join('results', 'generation', material)
  files = os.listdir(directory)
  files = [f for f in files if f.startswith(model)]

  if files == []:
    print(f'No files found for {model} in {material}')
    return

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

    # remove columns that contain the work 'white' or 'black'
    columns = [c for c in columns if 'White' not in c and 'Black' not in c]

    trainColumns = [c for c in columns if 'train' in c]
    testColumns = [c for c in columns if 'test' in c]

    dfs = []

    dfs.append(getColumns(data, columns, metaColumns, trainColumns, 'train'))
    dfs.append(getColumns(data, columns, metaColumns, testColumns, 'test'))

    
    # Check that the value metric is in at least one of the strings in the
    # train or test columns
    isIn = False
    for trainColumn in trainColumns:
      if metric in trainColumn:
        isIn = True
        break

    if not isIn:
      print(f'{metric} not in {modelID}')
      return

    isIn = False
    for testColumn in testColumns:
      if metric in testColumn:
        isIn = True
        break

    if not isIn:
      print(f'{metric} not in {modelID}')
      return

  
    # reset the index of the dataframes and concatenate them
    for df in dfs:
      df.reset_index(drop=True)
    data = pd.concat(dfs).reset_index(drop=True)

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

def getColumns(data, columns, metaColumns, modeColumns, mode):
    if len(modeColumns) // 4 > 1:
      subNetworks = []
      for colName in modeColumns:
        subNetwork = colName.split(' ')[1]
        if subNetwork not in subNetworks:
          subNetworks.append(subNetwork)
      
      for subNetwork in subNetworks:
        subNetworkColumns = [c for c in columns if f'{mode} {subNetwork}' in c]
        # Create a new dataframe with the data from data in the columns stated in
        # trainColumns and metaColumns
        subNetworkData = data[metaColumns + subNetworkColumns]
        subNetworkData['mode'] = f'{subNetwork} {mode}'
        subNetworkData = subNetworkData.rename(columns={c: formatMetricName(c) for c in subNetworkColumns})
        return subNetworkData.reset_index(drop=True)
    else:
      # Create a new dataframe with the data from data in the columns stated in
      # testColumns and metaColumns
      modeData = data[metaColumns + modeColumns]
      modeData['mode'] = mode
      modeData = modeData.rename(columns={c: formatMetricName(c) for c in modeColumns})
      return modeData.reset_index(drop=True)

if __name__ == '__main__':
  models = ['NN', 'CNN', 'GAN', 'DCGAN', 'VAE', 'ResNet']
  metrics = ['BCE', 'BCELogits', 'MSE', 'MAE', 'SSIM']
  materials = ['copper', 'aluminium']

  for material in materials:
    for model in models:
      for metric in metrics:
        plotPerformanceOverEpoch(
          material=material,
          model=model,
          metric=metric
        )