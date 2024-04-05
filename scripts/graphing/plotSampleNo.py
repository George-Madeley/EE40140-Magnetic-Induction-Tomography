import pandas as pd
import matplotlib.pyplot as plt

def plot_sample_counts():
  # Read the CSV file
  data = pd.read_csv('./data/data_samples.csv')

  # Make sure the 'sample' column is of type string
  data['sample'] = data['sample'].astype(str)

  # Group records by the 'sample' column and count the occurrences
  sample_counts = data.groupby('sample').size()

  # Plot the bar chart
  colors = ['darkgrey' if label < 'H' else 'orange' for label in sample_counts.index]
  plt.bar(sample_counts.index, sample_counts.values, color=colors)
  plt.xlabel('Sample')
  plt.ylabel('Count')
  plt.title('Sample Counts')
  plt.show()

if __name__ == '__main__':
  # Call the function to generate the plot
  plot_sample_counts()