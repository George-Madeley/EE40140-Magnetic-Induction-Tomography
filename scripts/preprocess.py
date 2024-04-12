import os

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import cleanUp
from preprocessing import formatting
from preprocessing.filters import colourFiltering, morphologicalFiltering
from preprocessing.utils import getCommonFactors, checkForProcessedFactors


def preprocessAllImages() -> None:
  
  """
  Preprocesses all images in the dataset.

  This function performs the following steps:
  1. Cleans up the dataset using the `cleanUp.main()` function.
  2. Calculates the common factors of the image width and height.
  3. Iterates over each common factor and processes the images.
  4. Creates a directory for each processed image size.
  5. For each image in the dataset, preprocesses the image and saves it in the corresponding directory.

  Returns:
      None
  """
  df = cleanUp.cleanUp()

  imageWidth: int = 640
  imageHeight: int = 480
  commonFactors: List[int] = getCommonFactors(imageWidth, imageHeight)


  for idx in range(len(commonFactors)):
    factor: int = commonFactors[idx]
    nextFactor: int = commonFactors[idx +
                                    1] if idx + 1 < len(commonFactors) else None

    print(f'Processing images with factor {factor}')

    if checkForProcessedFactors(nextFactor, imageWidth, imageHeight):
      continue

    newWidth: int = imageWidth // factor
    newHeight: int = imageHeight // factor

    directory: str = os.path.join(
        os.getcwd(),
        'images',
        'processed',
        f'{newHeight}x{newWidth}')
    os.makedirs(directory, exist_ok=True)

    for i, row in df.iterrows():
      # Get the filename
      bb_filename: str = row['bb_filename']
      cc_filename: str = row['cc_filename']
      sample: str = row['sample']

      savePath: str = os.path.join(directory, cc_filename)

      if os.path.exists(savePath):
        continue

      print(
        f'Processing image {cc_filename} to {newHeight}x{newWidth} sample: {sample}, i: {i}')

      # Preprocess the image
      image: np.ndarray = Preprocess(
          bb_filename, cc_filename, downsampleFactor=factor)

      # save image
      plt.imsave(savePath, image, cmap='gray')

  df = formatting.normalise(df)

  material_dfs = formatting.splitDataframeMaterial(df)

  for idx, material_df in enumerate(material_dfs):
    material = 'iron' if idx == 0 else 'copper'
    feature = 'shape' if idx == 0 else 'sample'
    material_df = formatting.getEvenDistribution(material_df, distFeature=feature)
    material_df = formatting.splitDataframeFeature(material_df, feature)
    material_df = formatting.oneHotEncode(material_df, feature)
    material_df.to_csv(f'./data/data_samples_{material}.csv', index=False)

  print('Preprocessing complete.')

def Preprocess(
  bb_filename: str,
  cc_filename: str,
  kernelSize: int = 5,
  cutoff: float = 0.5,
        downsampleFactor: int = 1) -> np.ndarray:
  """
  Preprocesses an image by performing various operations such as greyscale conversion, binary conversion,
  background removal, morphological filtering, and downsampling.

  Args:
      bb_filename (str): The filename of the background image used for background removal.
      cc_filename (str): The filename of the input image to be preprocessed.
      kernelSize (int, optional): The size of the kernel used for morphological filtering. Defaults to 5.
      cutoff (float, optional): The cutoff value used for binary conversion. Defaults to 0.5.
      downsampleFactor (int, optional): The factor by which the image is downsampled. Defaults to 1.

  Returns:
      numpy.ndarray: The preprocessed image.
  """

  imageDirectory: str = './images/original'

  # get the filename
  imagePath: str = os.path.join(imageDirectory, cc_filename)

  # get image from file
  image: np.ndarray = np.array(plt.imread(imagePath))

  # convert to greyscale
  image = colourFiltering.toGreyscale(image)

  # convert to binary
  image = colourFiltering.toBinary(image, cutoff)

  # remove background
  image = colourFiltering.removeBackground(
      image, f'{imageDirectory}/{bb_filename}')

  # convert to binary
  image = colourFiltering.toBinary(image, cutoff)

  # apply opening
  image = morphologicalFiltering.opening(image, kernelSize)

  # apply closing
  image = morphologicalFiltering.closing(image, kernelSize)

  # downsample
  image = colourFiltering.downSample(image, downsampleFactor)

  return image


if __name__ == '__main__':
  preprocessAllImages()
