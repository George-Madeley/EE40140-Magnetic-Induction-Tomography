from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

import os


def plotBestGenerator(
  metrics: list[str] = [
        'BCE Loss',
        'BCELogits Loss',
        'MSE Loss',
        'MAE Loss',
        'SSIM Loss'],
  verbose: bool = False) -> None:
  """
  Plots a bar graph of the best generator for each metric.

  Args:
    metrics (list[str], optional): List of metrics to plot. Defaults to
    ['BCE Loss', 'BCELogits Loss', 'MSE Loss', 'MAE Loss', 'SSIM Loss'].
    verbose (bool, optional): Whether to display the plot. Defaults to False.
  """
  file_path = os.path.join(
      'results',
      'generation',
      'best',
      'Best Variations.csv')
  df = pd.read_csv(file_path)

  for metric in metrics:
    # plot a bar graph of the best generator for each metric
    plt.figure(figsize=(21, 18))
    sns.set(font='Arial')
    sns.barplot(x='Model Name', y=metric, data=df, palette='magma')
    plt.title(f'Best Generator for {metric}', fontsize=50)
    plt.xlabel('Model Name', fontsize=45)
    plt.ylabel(metric, fontsize=45)

    # set font size of tick labels
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)

    save_path = os.path.join('images', 'graphs', 'best generator')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{metric}.png'))
    if verbose:
      plt.show()
    plt.close()


if __name__ == '__main__':
  plotBestGenerator()
