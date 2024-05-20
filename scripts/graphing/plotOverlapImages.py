import os
import numpy as np
from typing import List, Tuple
from utils import getTerminalArgs, formatString

import matplotlib.pyplot as plt


def overlapImages(
  down_scale_factor: int = 32,
  verbose: bool = False,
  cmap: str = 'viridis',
) -> None:
  """
  Generate and save an overlap image based on processed images.

  Args:
    down_scale_factor (int): The down scale factor for image dimensions. Default is 32.
    verbose (bool): Whether to display the generated image. Default is False.
    cmap (str): The colormap to use for the image. Default is 'viridis'.
  """
  width: int = 640 // down_scale_factor
  height: int = 480 // down_scale_factor

  image_dir: str = os.path.join(
    'images',
    'processed',
    f'{height}x{width}'
  )

  images: List[str] = os.listdir(image_dir)

  # Set the alpha value for each image
  alpha: float = 1 / len(images)

  fig, ax = plt.subplots()

  total_img: np.ndarray = np.zeros((height, width))

  for image_file in images:
    image_path: str = os.path.join(image_dir, image_file)
    rgb_img: np.ndarray = np.array(plt.imread(image_path))
    bw_img: np.ndarray = 0.2126 * rgb_img[:, :, 0] + 0.7152 * \
        rgb_img[:, :, 1] + 0.0722 * rgb_img[:, :, 2]
    bw_img = 1 - bw_img
    total_img += bw_img * alpha

  im = ax.imshow(total_img, cmap=cmap)
  cbar = fig.colorbar(im)
  cbar.set_label(formatString('Probability of Pixel Occupancy'))
  plt.axis('off')
  plt.title(formatString(
    f'Distribution of {len(images)} images\n{height}x{width} pixels'))
  plt.tight_layout()

  if verbose:
    plt.show()

  save_dir: str = os.path.join('images', 'graphs', 'overlap')
  os.makedirs(save_dir, exist_ok=True)
  save_path: str = os.path.join(save_dir, f'overlap - {height}x{width}.png')
  plt.savefig(save_path)


if __name__ == '__main__':
  args: Tuple[int, bool, str] = getTerminalArgs(['int', 'bool', 'str'])
  overlapImages(*args)
