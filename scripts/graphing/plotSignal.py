from typing import List
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt


def plotSignal(
  idx: int = None,
  verbose: bool = False,
) -> None:
  """
  Plots the recorded voltage differences for a given index.

  Args:
    idx (int): The index of the record to plot. If None, a random index will be generated.
    verbose (bool): If True, additional information will be printed and the plot will be displayed.
  """
  file_path: str = os.path.join('data', 'data_samples.csv')
  data: pd.DataFrame = pd.read_csv(file_path)

  if idx is None or idx >= data.shape[0] or idx < 0:
    # generate random index
    idx = np.random.randint(0, data.shape[0])

  # get the columns that start with 'cc_'
  cc_columns: List[str] = [col for col in data.columns if 'cc_' in col]

  # remove the column 'cc_filename'
  cc_columns.remove('cc_filename')

  # get the record at the index
  record: pd.Series = data.iloc[idx]

  # get the signal values
  signal: np.ndarray = record[cc_columns].values

  # plot the signal
  plt.figure(figsize=(8, 6))
  plt.plot(signal)
  plt.title(f'Recorded Voltage Differences from Index: {idx}', fontsize=20)
  plt.xlabel('Coil Combianation', fontsize=18)
  plt.ylabel('Voltage Difference (V)', fontsize=18)
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)

  save_path: str = os.path.join('images', 'graphs', 'signals')
  os.makedirs(save_path, exist_ok=True)
  plt.savefig(os.path.join(save_path, f"signal_{idx}.png"))

  if verbose:
    print(f"Index: {idx}")
    print(f"Filename: {record['cc_filename']}")
    plt.show()

  plt.close()


if __name__ == '__main__':
  plotSignal(verbose=True)
