from typing import List
import pandas as pd
import os

import matplotlib.pyplot as plt


def plotShapeNum() -> None:
  """
  Plot the number of records for each shape in the dataset.

  Reads a CSV file containing data samples and plots a stacked bar chart
  showing the number of records for each shape. The data is grouped by the
  'sample' column and filtered for different conditions to obtain the counts
  for null samples, aluminium samples, and copper samples. The chart is saved
  as an image file.
  """
  # Read the CSV file
  data: pd.DataFrame = pd.read_csv(os.path.join('data', 'data_samples.csv'))

  # Make sure the 'sample' column is of type string
  data['sample'] = data['sample'].astype(str)

  # Group records by the 'sample' column and count the occurrences
  sample_counts: pd.Series = data.groupby('sample').size()

  # Define the colors for the stacked bars
  colors: List[str] = ['black', 'blue', 'orange']

  # Filter the data for samples with 'H' or copperer
  copper_samples: pd.DataFrame = data[data['sample'] >= 'H']
  copper_sample_counts: pd.Series = copper_samples.groupby('shape').size()

  # Filter the data for samples higher than 'H'
  aluminium_samples: pd.DataFrame = data[data['sample'] < 'H']

  # pop samples from aluminium samples that are '0'
  null_samples: pd.DataFrame = aluminium_samples[aluminium_samples['sample'] == '0']
  null_samples_counts: pd.Series = null_samples.groupby('shape').size()

  aluminium_samples = aluminium_samples[aluminium_samples['sample'] != '0']
  aluminium_sample_counts: pd.Series = aluminium_samples.groupby('shape').size()

  # Plot the stacked bar chart
  plt.bar(
      null_samples_counts.index,
      null_samples_counts,
      color=colors[0],
      label='Null Samples')
  plt.bar(
      aluminium_sample_counts.index,
      aluminium_sample_counts,
      color=colors[1],
      label='Aluminium Samples')
  plt.bar(
      copper_sample_counts.index,
      copper_sample_counts,
      bottom=aluminium_sample_counts,
      color=colors[2],
      label='Copper Samples')

  # Add labels and legend
  plt.xlabel('Shape')
  plt.ylabel('Number of Records')
  plt.title('Shape Counts')

  # plot a key
  plt.legend()

  # Save the plot to a file
  save_path: str = os.path.join('images', 'graphs', 'shape counts')
  os.makedirs(save_path, exist_ok=True)
  plt.savefig(
      os.path.join(
          save_path,
          'shape - counts.png'),
      dpi=300,
      bbox_inches='tight')
  plt.close()


if __name__ == '__main__':
  # Call the function to generate the plot
  plotShapeNum()
