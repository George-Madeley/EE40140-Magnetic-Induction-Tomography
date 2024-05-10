import os
import matplotlib.pyplot as plt
import numpy as np

from utils import getTerminalArgs, formatString


def overlapImages(
    down_scale_factor: int = 32,
    verbose: bool = False,
    cmap: str = 'viridis',
  ):
  width = 640 // down_scale_factor
  height = 480 // down_scale_factor



  image_dir = os.path.join(
      'images',
      'processed',
      f'{height}x{width}')

  images = os.listdir(image_dir)

  # Set the alpha value for each image
  alpha = 1 / len(images)

  fig, ax = plt.subplots()

  total_img = np.zeros((height, width))

  for image_file in images:
    image_path = os.path.join(image_dir, image_file)
    rgb_img = np.array(plt.imread(image_path))
    bw_img = 0.2126 * rgb_img[:, :, 0] + 0.7152 * \
        rgb_img[:, :, 1] + 0.0722 * rgb_img[:, :, 2]
    bw_img = 1 - bw_img
    total_img += bw_img * alpha

  im = ax.imshow(total_img, cmap=cmap)
  cbar = fig.colorbar(im)
  cbar.set_label(formatString('Probability of Pixel Occupancy'))
  plt.axis('off')
  plt.title(formatString(f'Distribution of {len(images)} images\n{height}x{width} pixels'))
  plt.tight_layout()

  if verbose:
    plt.show()

  save_dir = os.path.join('images', 'graphs', 'overlap')
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, f'overlap - {height}x{width}.png')
  plt.savefig(save_path)


if __name__ == '__main__':
  args = getTerminalArgs(['int', 'bool', 'str'])
  overlapImages(*args)
