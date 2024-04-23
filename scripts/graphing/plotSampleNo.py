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
  colors = ['blue' if label < 'H' else 'orange' for label in sample_counts.index]
  plt.bar(sample_counts.index, sample_counts.values, color=colors)
  plt.xlabel('Sample')
  plt.ylabel('Count')
  plt.title('Sample Counts')

  # plot a key
  plt.bar(0, 0, color='blue', label='Iron Samples')
  plt.bar(0, 0, color='orange', label='Copper Samples')
  plt.legend()
  
  # Save the plot to a file
  plt.savefig('./images/graphs/sample_counts.png', dpi=300, bbox_inches='tight')
  plt.close()


if __name__ == '__main__':
  # Call the function to generate the plot
  plot_sample_counts()