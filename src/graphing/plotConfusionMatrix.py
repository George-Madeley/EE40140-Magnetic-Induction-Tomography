from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

import os

def plot_confusion_matrix(
  font_size=20,
  verbose=False
) -> None:
  """
  Plot confusion matrix for each CSV file in the 'results/classification/ConfusionMatrics' directory.

  Args:
    font_size (int): The font size for the plot (default is 20).
    verbose (bool): If True, display the plot (default is False).
  """
  directory = os.path.join('results', 'classification', 'ConfusionMatrics')
  for filename in os.listdir(directory):
    if not filename.endswith('.csv'):
      continue

    df = pd.read_csv(os.path.join(directory, filename), index_col=0)

    plt.figure(figsize=(5, 4))
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues')
    title = f'Confusion Matrix for {filename.split("-")[0]}'
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if verbose:
      plt.show()

    save_path = os.path.join('images', 'graphs', 'confusion matrix')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename.replace('.csv', '.png')))
    plt.close()


if __name__ == '__main__':
  plot_confusion_matrix()