from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os


def plotPerformanceByArea(
  material: str = 'iron',
  verbose: bool = False,
  down_scale_factor: int = 32,
  metric: str = 'BCE'
) -> None:
  """
  Plots the performance of a given material by area.

  Args:
    material (str): The material to plot the performance for. Default is 'iron'.
    verbose (bool): Whether to display the plot. Default is False.
    down_scale_factor (int): The down scale factor for image size. Default is 32.
    metric (str): The performance metric to use. Default is 'BCE'.
  """
  width: int = 640 // down_scale_factor
  height: int = 480 // down_scale_factor

  directory: str = os.path.join('results', 'generation', material, 'per image')
  files: List[str] = os.listdir(directory)
  files = [f for f in files if f.endswith('.csv')]
  files = [os.path.join(directory, f) for f in files]
  df_performance: pd.DataFrame = pd.concat([pd.read_csv(f) for f in files])

  sample_material: str = 'iron' if material == 'aluminium' else material

  data_path: str = os.path.join('data', f'data_samples_{sample_material}.csv')
  df_data: pd.DataFrame = pd.read_csv(data_path)

  groups: pd.core.groupby.generic.DataFrameGroupBy = df_performance.groupby(
    'Label')
  for label, group in groups:
    indices: pd.Series = group['Index']
    performace: pd.Series = group[metric]

    # Get the cc_filenames from the data for the indices
    cc_filenames: pd.Series = df_data.loc[indices, 'cc_filename']

    img_dir: str = os.path.join('images', 'processed', f'{height}x{width}')

    alpha: float = 1 / len(cc_filenames)

    total_img: np.ndarray = np.zeros((height, width))
    images: np.ndarray = np.zeros((len(cc_filenames), 1, height, width))

    for idx, image_file in enumerate(cc_filenames):
      image_path: str = os.path.join(img_dir, image_file)
      rgb_img: np.ndarray = np.array(plt.imread(image_path))

      r: np.ndarray = rgb_img[:, :, 0]
      g: np.ndarray = rgb_img[:, :, 1]
      b: np.ndarray = rgb_img[:, :, 2]
      bw_img: np.ndarray = 0.2126 * r + 0.7152 * g + 0.0722 * b
      bw_img = 1 - bw_img

      images[idx, 0, :, :] = bw_img

      total_img += bw_img * alpha

    images = np.multiply(images, performace.values.reshape(-1, 1, 1, 1))

    bias: np.ndarray = total_img

    # images = images / bias

    # replace NaN with 0
    images = np.nan_to_num(images)

    images = np.mean(images, axis=0)

    diff1: int = (width - height) // 2
    diff2: int = ((width - height) // 2) + 1
    images = images[0][:, diff1:width - diff2]

    # find the min value that is not 0
    nonzeros_location: Tuple[np.ndarray, np.ndarray] = np.nonzero(images)
    if nonzeros_location[0].size == 0 and nonzeros_location[1].size == 0:
      min_val: float = np.min(images)
    else:
      min_val = np.min(images[np.nonzero(images)])

    # replace 0 with the min value
    images[images == 0] = min_val

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    im = ax.imshow(images, cmap='viridis')
    cbar = fig.colorbar(im)
    cbar.set_label('Probability of Lower MAE')
    plt.axis('off')
    plt.title(f'{label} - {metric} - {height}x{width} pixels')
    plt.tight_layout()

    if verbose:
      plt.show()

    save_dir: str = os.path.join('images', 'graphs', 'performance by area')
    os.makedirs(save_dir, exist_ok=True)
    save_path: str = os.path.join(
        save_dir, f'{label} - {metric} - {down_scale_factor} - {height}x{width}.png')
    plt.savefig(save_path)


if __name__ == '__main__':
  plotPerformanceByArea(
    down_scale_factor=8,
    material='copper',
    metric='MAE'
  )
