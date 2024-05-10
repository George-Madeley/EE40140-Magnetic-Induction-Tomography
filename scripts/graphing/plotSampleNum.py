import pandas as pd
import matplotlib.pyplot as plt

import os

def plotSampleNum(
    verbose: bool = False
):
  # Read the CSV file
  file_path = os.path.join('data', 'data_samples.csv')
  data = pd.read_csv(file_path)

  # Make sure the 'sample' column is of type string
  data['sample'] = data['sample'].astype(str)

  # Group records by the 'sample' column and count the occurrences
  sample_counts = data.groupby('sample').size()

  # Plot the bar chart
  colors = ['blue' if label < 'H' else 'orange' for label in sample_counts.index]
  colors[0] = 'black'
  plt.bar(sample_counts.index, sample_counts.values, color=colors)
  plt.xlabel('Sample')
  plt.ylabel('Count')
  plt.title('Sample Counts')

  # plot a key
  plt.bar(0, 0, color='black', label='Null Samples')
  plt.bar(0, 0, color='blue', label='Aluminium Samples')
  plt.bar(0, 0, color='orange', label='Copper Samples')
  plt.legend()
  
  # Save the plot to a file
  save_path = os.path.join('images', 'graphs', 'sample counts')
  os.makedirs(save_path, exist_ok=True)
  plt.savefig(os.path.join(save_path, 'sample_counts.png'), dpi=300, bbox_inches='tight')

  if verbose:
    plt.show()

  plt.close()


if __name__ == '__main__':
  # Call the function to generate the plot
  plotSampleNum()