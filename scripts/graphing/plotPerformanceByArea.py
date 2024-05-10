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
):
  width = 640 // down_scale_factor
  height = 480 // down_scale_factor


  directory = os.path.join('results', 'generation', material, 'per image')
  files = os.listdir(directory)
  files = [f for f in files if f.endswith('.csv')]
  files = [os.path.join(directory, f) for f in files]
  df_performance = pd.concat([pd.read_csv(f) for f in files])

  sample_material = 'iron' if material == 'aluminium' else material

  data_path = os.path.join('data', f'data_samples_{sample_material}.csv')
  df_data = pd.read_csv(data_path)

  groups = df_performance.groupby('Label')
  for label, group in groups:
    indices = group['Index']
    performace = group[metric]

    # Get the cc_filenames from the data for the indices
    cc_filenames = df_data.loc[indices, 'cc_filename']

    img_dir = os.path.join('images', 'processed', f'{height}x{width}')

    alpha = 1 / len(cc_filenames)

    total_img = np.zeros((height, width))
    images = np.zeros((len(cc_filenames), 1, height, width))

    for idx, image_file in enumerate(cc_filenames):
      image_path = os.path.join(img_dir, image_file)
      rgb_img = np.array(plt.imread(image_path))

      r = rgb_img[:, :, 0]
      g = rgb_img[:, :, 1]
      b = rgb_img[:, :, 2]
      bw_img = 0.2126 * r + 0.7152 * g + 0.0722 * b
      bw_img = 1 - bw_img

      images[idx, 0, :, :] = bw_img

      total_img += bw_img * alpha

    images = np.multiply(images, performace.values.reshape(-1, 1, 1, 1))

    bias = total_img

    # images = images / bias

    # replace NaN with 0
    images = np.nan_to_num(images)

    images = np.mean(images, axis=0)

    diff1 = (width - height) // 2
    diff2 = ((width - height) // 2) + 1
    images = images[0][:, diff1:width - diff2]

    # find the min value that is not 0
    nonzeros_location = np.nonzero(images)
    if nonzeros_location[0].size == 0 and nonzeros_location[1].size == 0:
      min_val = np.min(images)
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

    save_dir = os.path.join('images', 'graphs', 'performance by area')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{label} - {metric} - {down_scale_factor} - {height}x{width}.png')
    plt.savefig(save_path)

if __name__ == '__main__':
  plotPerformanceByArea(
    down_scale_factor=8,
    material='copper',
    metric='MAE'
  )
  