import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import formatString

import os


def plotBestClassification(
  verbose: bool = False,
) -> None:
  """
  Plot the best classification results for each model. Does four plots for each
  model. Two for the material (aluminum and copper) and two for the number of
  samples (120 and 240).

  Args:
    verbose (bool, optional): If True, display the plot. Defaults to False.
  """
  models = [
    'DecisionTree',
    'KNearestNeighbors',
    'NearestCentroid',
    'RandomForest',
    'StochasticGradientDescent',
    'SupportVectorMachine'
  ]

  df_best = pd.DataFrame()

  for model in models:
    directory = os.path.join('results', 'classification')
    files = os.listdir(directory)
    files = [f for f in files if f.startswith(model)]
    dfs = [pd.read_csv(os.path.join(directory, f)) for f in files]
    data = pd.concat(dfs)

    # replace the material 'iron' with 'aluminum'
    data['material'] = data['material'].replace('iron', 'aluminum')

    # remove any record where the accuracy is 1.0
    data = data[data['Accuracy'] < 1.0]

    # Get the list of columns between 'material' and 'accuracy'
    columns = data.columns.tolist()
    model_index = columns.index('numSamples')
    accuracy_index = columns.index('Accuracy')
    columns = columns[model_index + 1:accuracy_index]

    # group by numSamples
    groups = data.groupby(['numSamples', 'material'])

    for name, group in groups:

      # Reset the index of the group DataFrame
      group = group.reset_index(drop=True)

      # Drop the columns that are not needed from the group
      group = group.drop(columns, axis=1)

      best = group.loc[group['Accuracy'].idxmax()]

      # Convert the best record to a DataFrame
      best = pd.DataFrame(best).T

      # Concatenate the best record to the best DataFrame
      df_best = pd.concat([df_best, best])

  # group the best DataFrame by the material
  groups = df_best.groupby('material')

  for material, group in groups:

    # Format the model column by passing each value to the formatString function
    group['model'] = group['model'].apply(formatString)

    # save the group to a csv file
    save_dir = os.path.join('results', 'classification', 'best')
    os.makedirs(save_dir, exist_ok=True)
    group.to_csv(os.path.join(save_dir, f'best-{material}.csv'), index=False)

  fontSize = 20

  # Plot the best classification results by creating a bar chart where the
  # x-axis is the model and the y-axis is the accuracy and the hue is the
  # numSamples
  sns.set_theme(style='whitegrid')
  sns.set_context('talk')

  plt.figure(figsize=(10, 7))
  sns.barplot(data=df_best, x='model', y='Accuracy', hue='numSamples')
  plt.title(
    f'Best Classification Results for each Model',
    fontsize=fontSize + 4)
  plt.ylabel('Accuracy', fontsize=fontSize + 2)
  plt.xlabel('Model', fontsize=fontSize + 2)
  plt.gca().set_ylim([0.6, 1.0])
  plt.yticks(fontsize=fontSize)
  plt.gca().set_xticklabels(
    [formatString(label.get_text()) for label in plt.gca().get_xticklabels()],
    fontsize=fontSize,
  )
  plt.legend(title='Number of Samples', loc='lower right')

  save_path = os.path.join('images', 'graphs', 'best classification')
  os.makedirs(save_path, exist_ok=True)
  plt.savefig(os.path.join(save_path, f'best.png'))

  if verbose:
    plt.show()

  plt.close()


if __name__ == '__main__':
  plotBestClassification()
