import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os

def plotBestClassification():

  models = [
      'DecisionTree',
      'KNearestNeighbors',
      'NearestCentroid',
      'RandomForest',
      'StochasticGradientDescent',
      'SupportVectorMachine'
  ]

  df_best = pd.DataFrame(columns=[
    'model',
    'material',
    'numSamples',
    'Accuracy',
    'F1',
    'Precision',
    'Recall',
    'mean_fit_time',
    'std_fit_time',
    'mean_score_time',
    'std_score_time'
  ])

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
    data = data.groupby(['numSamples', 'material'])

    for name, group in data:

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
  grouped = df_best.groupby('material')

  for material, group in grouped:

    # Format the model column by passing each value to the formatString function
    group['model'] = group['model'].apply(formatString)

    # save the group to a csv file
    group.to_csv(os.path.join('results', 'classification', f'best-{material}.csv'), index=False)

  fontSize = 20

  # Plot the best classification results by creating a bar chart where the
  # x-axis is the model and the y-axis is the accuracy and the hue is the
  # numSamples
  sns.set_theme(style='whitegrid')
  sns.set_context('talk')

  plt.figure(figsize=(10, 7))
  sns.barplot(data=df_best, x='model', y='Accuracy', hue='numSamples')
  plt.title(f'Best Classification Results for each Model', fontsize=fontSize+4)
  plt.ylabel('Accuracy', fontsize=fontSize+2)
  plt.xlabel('Model', fontsize=fontSize+2)
  plt.gca().set_ylim([0.6, 1.0])
  plt.yticks(fontsize=fontSize)
  plt.gca().set_xticklabels(
    [formatString(label.get_text()) for label in plt.gca().get_xticklabels()],
    fontsize=fontSize,
  )
  plt.legend(title='Number of Samples', loc='lower right')

  plt.savefig(os.path.join('images', 'graphs', f'best.png'))

  plt.show()

  plt.close()


def formatString(string: str) -> str:
  """
  Formats a string to be more readable.

  Args:
      string (str): The string to format.

  Returns:
      str: The formatted string.
  """

  if string == 'iron':
    string = 'aluminum'

  # if there is a captial letter in the string that is not the first letter
  # add a space before it
  modelInitials = {
    'DecisionTree': 'DT',
    'KNearestNeighbors': 'KNN',
    'NearestCentroid': 'NC',
    'NeuralNetwork': 'NN',
    'RandomForest': 'RF',
    'StochasticGradientDescent': 'SGD',
    'SupportVectorMachine': 'SVM'
  }
  if string in modelInitials.keys():
    return modelInitials[string]


  for i in range(1, len(string)):
    if string[i].isupper() and string[i - 1] != ' ':
      string = string[:i] + ' ' + string[i:]
  return string.replace('_', ' ').title()

if __name__ == '__main__':
  plotBestClassification()

