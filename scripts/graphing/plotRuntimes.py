from typing import List
import seaborn as sns
import pandas as pd
import os

import matplotlib.pyplot as plt


def plotRuntimes(verbose: bool = False) -> None:
  """
  Plots the runtimes from a CSV file against the downscale factor.

  Args:
    verbose (bool, optional): If True, displays the plot for each time column. Defaults to False.
  """
  file_path: str = os.path.join('results', 'generation', 'runtimes.csv')
  df: pd.DataFrame = pd.read_csv(file_path)

  # all columns should be floats
  df = df.astype(float)

  # get the columns that are not 'Downscale Factor'
  time_columns: List[str] = df.columns[1:]

  # multiple the data by 1e-9 to convert from ns to s
  df[time_columns] *= 1e-9

  # replace (ns) with (s) in the column names
  df.columns = df.columns.str.replace(' (ns)', ' (s)')

  time_columns = [column.replace(' (ns)', ' (s)') for column in time_columns]

  directory: str = os.path.join('images', 'graphs', 'runtimes')
  os.makedirs(directory, exist_ok=True)

  for time_column in time_columns:
    # plot a line graph of the time column vs the downscale factor
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x='Downscale Factor', y=time_column)
    plt.title(f'{time_column} vs Downscale Factor')
    plt.xlabel('Downscale Factor')
    plt.ylabel(time_column)
    plt.xscale('log', base=2)
    plt.xlim(0)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    save_path: str = os.path.join('images', 'graphs', 'runtimes')
    plt.savefig(os.path.join(save_path, f'{time_column}.png'))

    if verbose:
      plt.show()

    plt.close()


if __name__ == '__main__':
  plotRuntimes(verbose=True)
