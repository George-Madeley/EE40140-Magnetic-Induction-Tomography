import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os


def plotHeatMapModelMetric(
    specific_models: list[str] = None,
    material: str = 'aluminium',
    useColor: bool = True,
    verbose: bool = False
):
  directory = os.path.join('results', 'generation', 'best')
  files = os.listdir(directory)
  files = [f for f in files if f.endswith('.csv') and material in f]

  # Remove 'Best Variations.csv' from the files
  files = [f for f in files if 'Best Variations' not in f]

  if material == 'aluminium':
    labels = ['X', 'T', 'S', '_', 'N']
  else:
    labels = ['0', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

  df = pd.DataFrame()

  for f in files:
    if 'ResNet' in f:
      continue

    data = pd.read_csv(os.path.join(directory, f))

    if 'GAN' in f or 'DCGAN' in f:
      # remove any columns that start with 'D '
      data = data[[c for c in data.columns if not c.startswith('D ')]]

    # find any columns that begin with '  ' and rename them to '_ '
    for c in data.columns:
      if c.startswith('  '):
        data = data.rename(columns={c: f'_ {c[2:]}'})

    if specific_models is not None:
      data = data[data['Model Name'].isin(specific_models)]
    else:
      # get the record with the minimum MAE Loss
      data = data[data['MAE Loss'] == data['MAE Loss'].min()]

    df = pd.concat([df, data])

  if useColor:
    useColorMetrics(verbose, labels, df, material)
  else:
    meanColorMetrics(verbose, df, material)


def meanColorMetrics(verbose, df, material):
  keep_columns = ['Model Name']

  white_columns = [c for c in df.columns if 'White' in c]
  white_df = df[keep_columns + white_columns]
  # Remove 'White ' from the column names
  white_df.columns = [c.replace('White ', '') for c in white_df.columns]

  black_columns = [c for c in df.columns if 'Black' in c]
  black_df = df[keep_columns + black_columns]
  # Remove 'Black ' from the column names
  black_df.columns = [c.replace('Black ', '') for c in black_df.columns]

  df = pd.concat([white_df, black_df])

  # group by the 'Model Name' and take the mean of the columns
  df = df.groupby('Model Name').mean().reset_index()

  new_columns = ['Model Name'] + [c.split(' ')[0] for c in df.columns[1:]]
  df.columns = new_columns

  # create a heatmap of the data where the rows are the 'Model Name' and the columns are the columns
  # that contain the color
  plt.figure(figsize=(8, 5))
  sns.heatmap(
      df.set_index('Model Name').T,
      annot=True,
      fmt=".2e",
      cmap='coolwarm')
  plt.title('MAE Heatmap of Best Models')
  plt.xlabel('Model Name')
  plt.ylabel('MAE Loss')
  plt.yticks(rotation=0)
  plt.tight_layout()

  save_path = os.path.join('images', 'graphs', 'heatmap model metric')
  os.makedirs(save_path, exist_ok=True)
  plt.savefig(os.path.join(save_path, f'{material} - MAE Heatmap.png'))

  if verbose:
    plt.show()

  plt.close()


def useColorMetrics(verbose, labels, df, material):
  colors = ['Black', 'White']

  for color in colors:
    # get the 'Model Name' and and columns that contain the color from df
    keep_columns = ['Model Name']
    for label in labels:
      keep_columns += [c for c in df.columns if color in c and c.startswith(label)]

    colorDf = df[keep_columns]

    # Rename the columns
    new_columns = ['Model Name'] + \
        [c.split(' ')[0] for c in colorDf.columns[1:]]
    colorDf.columns = new_columns

    if color == 'Black':
      try:
        colorDf = colorDf.drop('_', axis=1)
      except KeyError:
        pass

      try:
        colorDf = colorDf.drop('0', axis=1)
      except KeyError:
        pass

    # create a heatmap of the data where the rows are the 'Model Name' and the columns are the columns
    # that contain the color
    font_size = 14
    plt.figure(figsize=(8, 5))
    sns.heatmap(
      colorDf.set_index('Model Name').T,
      annot=True,
      fmt=".2e",
      cmap='coolwarm',
      annot_kws={"fontsize": font_size})
    plt.title(f'{color} MAE Heatmap of Best Models', fontsize=font_size + 4)
    plt.xlabel('Model Name', fontsize=font_size + 2)
    plt.ylabel('MAE Loss', fontsize=font_size + 2)
    plt.yticks(rotation=0, fontsize=font_size)
    plt.xticks(rotation=0, fontsize=font_size)
    plt.tight_layout()

    

    save_path = os.path.join('images', 'graphs', 'heatmap')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{material} - {color} MAE Heatmap.png'))

    if verbose:
      plt.show()

    plt.close()


if __name__ == '__main__':
  plotHeatMapModelMetric(
      specific_models=[
          'CNN6',
          'DCGAN4',
          'GAN7',
          'NN7',
          'ResNet1',
          'VAE4',
          'UNN1'],
      material='copper',
      useColor=True,
      verbose=True)
