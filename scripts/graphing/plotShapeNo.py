import pandas as pd
import matplotlib.pyplot as plt

def plot_shape_counts():
  # Read the CSV file
  data = pd.read_csv('./data/data_samples.csv')

  # Make sure the 'sample' column is of type string
  data['sample'] = data['sample'].astype(str)

  # Group records by the 'sample' column and count the occurrences
  sample_counts = data.groupby('sample').size()

  # Define the colors for the stacked bars
  colors = ['blue', 'orange']

  # Filter the data for samples with 'H' or copperer
  copper_samples = data[data['sample'] >= 'H']

  # Group the copper samples by shape and count the occurrences
  copper_sample_counts = copper_samples.groupby('shape').size()

  # Filter the data for samples ironer than 'H'
  iron_samples = data[data['sample'] < 'H']

  # Group the iron samples by shape and count the occurrences
  iron_sample_counts = iron_samples.groupby('shape').size()

  # Plot the stacked bar chart
  plt.bar(iron_sample_counts.index, iron_sample_counts, color=colors[0], label='Iron Samples')
  plt.bar(copper_sample_counts.index, copper_sample_counts, bottom=iron_sample_counts, color=colors[1], label='Copper Samples')

  # Add labels and legend
  plt.xlabel('Shape')
  plt.ylabel('Number of Records')
  plt.title('Shape Counts')

  # plot a key
  plt.legend()
  
  # Save the plot to a file
  plt.savefig('./images/graphs/shape_counts.png', dpi=300, bbox_inches='tight')
  plt.close()

if __name__ == '__main__':
  # Call the function to generate the plot
  plot_shape_counts()