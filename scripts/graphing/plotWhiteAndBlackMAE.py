import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def plotWhiteAndBlackMAE(
    file_path: str,
    model_name: str,
    epoch: int,
    material: str = 'aluminium',
    verbose: bool = False
):
  if not os.path.exists(file_path):
    print(f'File {file_path} does not exist')
    return
  
  data = pd.read_csv(file_path)

  # get the record for the specified epoch
  data = data[data['Epoch'] == epoch]

  # raise an error if there is more than one record for the specified epoch
  if len(data) != 1:
    raise ValueError(f'More than one record found for epoch {epoch}')
  
  # Get a list of the columns that contain 'White' or 'Black'
  whiteColumns = [c for c in data.columns if 'White' in c]
  blackColumns = [c for c in data.columns if 'Black' in c]
  
  df = pd.DataFrame(columns=['MAE', 'Label', 'Pixel'])
  for c in whiteColumns:
    label = c.split(' ')[2]
    white_df = pd.DataFrame({'MAE': data[c].values[0], 'Label': label, 'Pixel': 'White'}, index=[0])
    df = pd.concat([df, white_df], ignore_index=True)

  for c in blackColumns:
    label = c.split(' ')[2]
    black_df = pd.DataFrame({'MAE': data[c].values[0], 'Label': label, 'Pixel': 'Black'}, index=[0])
    df = pd.concat([df, black_df], ignore_index=True)

  # group the data by the label and pixel type and get the mean MAE
  df = df.groupby(['Label', 'Pixel']).mean().reset_index()

  # plot the data to a bar chart where the x-axis is the label and the y-axis is
  # the MAE. The hue is the pixel type
  plt.figure(figsize=(5, 3.5))
  sns.barplot(x='Label', y='MAE', hue='Pixel', data=df)
  plt.title(f'MAE for White and Black Pixels for {model_name}')
  plt.legend(title='Pixel Color')
  plt.xlabel('Label')
  plt.ylabel('MAE')
  plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Use scientific notation for y-axis
  plt.tight_layout()

  directory = os.path.join('images', 'graphs', 'white and black')
  os.makedirs(directory, exist_ok=True)
  file_path = f'{material}_{model_name}.png'
  file_path = os.path.join(directory, file_path)
  plt.savefig(file_path)

  if verbose:
    print(df)
    plt.show()

  plt.close()

if __name__ == '__main__':
  args = sys.argv[2:]
  file_name = args[0]
  model_name = args[1]
  epoch = int(args[2])
  verbose = args[3] == 'True'
  plotWhiteAndBlackMAE(file_name, model_name, epoch, verbose=verbose)

  
