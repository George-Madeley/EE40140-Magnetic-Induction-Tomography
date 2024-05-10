from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from utils import formatString

import os

def plotBestModelVersions(
    material: str = 'aluminium',
    epoch_limit: int = 1000,
    verbose: bool = False
):
  models = {
    'CNN': 'N',
    'DCGAN': 'G',
    'GAN': 'G',
    'NN': 'N',
    'VAE': 'VAE',
    'UNN': 'UNET',
    'ResNet': 'ResNet',
  }

  directory = os.path.join('results', 'generation', material)
  files = os.listdir(directory)

  for model, sub_model in models.items():
    model_files = [file for file in files if file.startswith(model)]
    df = pd.DataFrame()

    metric = f'test {sub_model} MAE Loss'

    for model_file in model_files:
      model_df = pd.read_csv(os.path.join(directory, model_file))

      model_id = model_file.split(' - ')[0]

      model_df['Model Name'] = model_id

      # Limit the number of epochs to the epoch limit
      model_df = model_df[model_df['Epoch'] <= epoch_limit]

      # Find the records in each group with the min metric
      model_df = model_df[model_df[metric] == model_df[metric].min()]
      model_df = model_df.reset_index(drop=True)
      df = pd.concat([df, model_df])

    # remove all the columns with 'train' in the name
    df = df[df.columns.drop(list(df.filter(regex='train')))]

    # replace all the columns with 'test' in the name with ''
    df.columns = df.columns.str.replace('test ', '')
    if sub_model != 'N':
      df.columns = df.columns.str.replace(f'{sub_model} ', '')
    else:
      # find the columns that start with 'N '
      for c in df.columns:
        if c.startswith('N '):
          df = df.rename(columns={c: c[2:]})

    # format all the column names
    df.columns = [formatString(col) for col in df.columns]

    # save the dataframe to a csv file
    save_dir = os.path.join('results', 'generation', 'best')
    os.makedirs(save_dir, exist_ok=True)
    savepath = os.path.join(save_dir, f'{model} {material} - best.csv')
    df.to_csv(savepath, index=False)

    # plot the dataframe
    sns.set(style='whitegrid')
    sns.set_context('talk')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model Name', y=f'MAE Loss')
    plt.xlabel('Model Name', fontsize=20)
    plt.ylabel(f'MAE Loss', fontsize=20)

    # Use scientific notation for the y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.title(f'{model} MAE Loss for {material} Models', fontsize=20)
    plt.tight_layout()

    save_dir = os.path.join('images', 'graphs', 'best-model-versions')
    os.makedirs(save_dir, exist_ok=True)
    savepath = os.path.join(save_dir, f'{model} {material} - best.png')
    plt.savefig(savepath)

    if verbose:
      plt.show()

    plt.close()


if __name__ == '__main__':
  plotBestModelVersions(epoch_limit=500, material='copper')