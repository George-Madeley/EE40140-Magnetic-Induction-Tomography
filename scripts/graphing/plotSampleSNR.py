import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_sample_snr():
  # Read the CSV file
  data = pd.read_csv('./data/data_samples.csv')

  # Make sure the 'sample' column is of type string
  data['sample'] = data['sample'].astype(str)

  # get the column names starting with 'bb_' and 'cc_'
  bb_columns = [col for col in data.columns if col.startswith('bb_')]
  cc_columns = [col for col in data.columns if col.startswith('cc_')]

  # remove 'bb_filename' and 'cc_filename' from the list of columns
  bb_columns.remove('bb_filename')
  cc_columns.remove('cc_filename')

  # set these columns to float
  data[bb_columns] = data[bb_columns].astype(float)
  data[cc_columns] = data[cc_columns].astype(float)

  # for each sample, get the cc_ values and minues the bb_ values.
  # then divide the result by the bb_values to get the SNR. Calculate the mean
  # of the SNR values for each group and calculate the log10 of the mean SNR
  # values
  diff = data[cc_columns].values - data[bb_columns].values
  absDiff = np.abs(diff)
  snr = absDiff / data[bb_columns]
  # replace infinite values with NaN
  snr = snr.replace([np.inf, -np.inf], 0)
  snr = snr.mean(axis=0)

  # Calculate the log of each value
  snr = 10 * np.log10(snr)

  # Plot a line chart of the SNR values for each group. The x-axis should be the
  # column names of the SNR data and the y-axis should be the SNR values
  plt.plot(snr.index, snr.values)
  plt.xlabel('Column')
  plt.ylabel('SNR (dB)')
  plt.title('Sample SNR')
  
  # Save the plot to a file
  plt.savefig('./images/graphs/sample_snr.png', dpi=300, bbox_inches='tight')
  plt.close()


if __name__ == '__main__':
  # Call the function to generate the plot
  plot_sample_snr()