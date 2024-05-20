import pandas as pd
import seaborn as sns
import os
import sys

import matplotlib.pyplot as plt


def plotGenerativeMetrics(
  specific_models: list[str] = None,
  metric: str = 'MAE',
  material: str = 'aluminium',
  verbose: bool = False
) -> None:
  """
  Plot generative metrics for different models.

  Args:
    specific_models (list[str], optional): List of specific models to include. Defaults to None.
    metric (str, optional): Metric to plot. Defaults to 'MAE'.
    material (str, optional): Material type. Defaults to 'aluminium'.
    verbose (bool, optional): Whether to display additional information. Defaults to False.
  """
  directory = os.path.join('results', 'generation', material, 'best')
  files = os.listdir(directory)
  files = [f for f in files if f.endswith('.csv')]

  df = pd.DataFrame()

  for f in files:
    data = pd.read_csv(os.path.join(directory, f))

    if 'GAN' in f or 'DCGAN' in f:
      # remove any columns that start with 'D '
      data = data[[c for c in data.columns if not c.startswith('D ')]]

    # Remove any columns that contain 'White' or 'Black'
    data = data[[c for c in data.columns if 'White' not in c and 'Black' not in c]]

    # find the column that contains 'S S I M' and rename it to 'SSIM'
    for c in data.columns:
      if 'S S I M' in c:
        data = data.rename(columns={c: c.replace('S S I M', 'SSIM')})

    if specific_models is not None:
      data = data[data['Model Name'].isin(specific_models)]
    else:
      # get the record with the minimum MAE Loss
      data = data[data[f'{metric} Loss'] == data[f'{metric} Loss'].min()]

    df = pd.concat([df, data])

  # check if the metric is in the columns
  if f'{metric} Loss' not in df.columns:
    print(f'{metric} Loss not in columns')
    return

  # plot a bar chart of the metric
  plt.figure(figsize=(5, 3.5))
  sns.barplot(x='Model Name', y=f'{metric} Loss', data=df)
  plt.title(f'{metric} for Generative Models')
  plt.xlabel('Model Name')
  plt.ylabel(f'{metric} Loss')
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  plt.tight_layout()

  save_path = os.path.join('images', 'graphs', 'generative metrics')
  os.makedirs(save_path, exist_ok=True)
  plt.savefig(os.path.join(save_path, f'{metric} Loss.png'))

  save_path = os.path.join('results', 'generation', 'best')
  os.makedirs(save_path, exist_ok=True)
  df.to_csv(os.path.join(save_path, f'Best Variations.csv'), index=False)

  if verbose:
    print(df)
    plt.show()

  plt.close()

if __name__ == '__main__':
  args = sys.argv[2:]
  metric = args[0]

  if len(args) == 1:
    verbose = False
  else:
    verbose = args[1] == 'True'

  plotGenerativeMetrics(
    specific_models=['CNN6', 'DCGAN4', 'GAN7', 'NN7', 'ResNet1', 'VAE4', 'UNN1'],
    metric=metric,
    verbose=verbose
  )