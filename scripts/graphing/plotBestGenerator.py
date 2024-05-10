from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

import os

def plotBestGenerator(
    metrics: list[str] = ['BCE Loss', 'BCELogits Loss', 'MSE Loss', 'MAE Loss', 'S S I M Loss'],
    verbose: bool = False
):
  directory = os.path.join('results', 'generation', 'best')
  files = os.listdir(directory)

  columns_to_keep = [
    'Model Name',
    'Learning Rate',
    'Down Scale Factor',
    'Batch Size',
    'Noise',
    'Epoch',
    'BCE Loss',
    'BCELogits Loss',
    'MSE Loss',
    'MAE Loss',
    'S S I M Loss',
  ]

  dfs = []

  for f in files:
    df = pd.read_csv(os.path.join(directory, f))
    # drop columns that are not in columns_to_keep
    df = df[df.columns.intersection(columns_to_keep)]
    # get the record with the lowest MAE Loss
    df = df[df['MAE Loss'] == df['MAE Loss'].min()]

    dfs.append(df)

  df = pd.concat(dfs)

  for metric in metrics:
    # plot a bar graph of the best generator for each metric
    plt.figure(figsize=(5, 4))
    sns.barplot(x='Model Name', y=metric, data=df)
    plt.title(f'Best Generator for {metric}')
    plt.xlabel('Model Name')
    plt.ylabel(metric)

    # use scientific notation for the y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    save_path = os.path.join('images', 'graphs', 'best_generator')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{metric}.png'))
    if verbose:
      plt.show()
    plt.close()

if __name__ == '__main__':
  plotBestGenerator()
    