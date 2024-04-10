import os

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from colourFiltering import ColourFiltering
from frequencyFiltering import FrequencyFiltering
from morphologicalFiltering import MorphologicalFiltering
from cleanUp import cleanUp

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
  df = cleanUp.main()

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

  df_iron, df_copper = cleanUp.splitDataframeMaterial(df)
  df_iron = cleanUp.getEvenDistribution(df_iron, distFeature='shape')
  df_iron = cleanUp.splitDataframe(df_iron, 'shape')
  df_copper = cleanUp.getEvenDistribution(df_copper, distFeature='sample')
  df_copper = cleanUp.splitDataframe(df_copper, 'sample')

  print('Preprocessing complete.')


def getCommonFactors(a: int, b: int, maxFactor: int = 32) -> List[int]:
  """
  Returns a list of common factors between two numbers within a specified range.

  Parameters:
  a (int): The first number.
  b (int): The second number.
  maxFactor (int, optional): The maximum factor to consider. Defaults to 32.

  Returns:
  list: A list of common factors between a and b.

  """
  factors: List[int] = []
  for i in range(1, min(a, b) + 1):
    if i > maxFactor:
      break

    if a % i == 0 and b % i == 0:
      factors.append(i)
  return factors


def checkForProcessedFactors(
  nextFactor: int,
  imageWidth: int,
        imageHeight: int) -> bool:
  """
  Checks if the images have already been processed for a given factor.

  Args:
      nextFactor (int): The next factor to be checked.
      imageWidth (int): The width of the original image.
      imageHeight (int): The height of the original image.

  Returns:
      bool: True if the images have already been processed for the given factor, False otherwise.
  """
  # If the next factor directory exists, then the images have already been
  # processed for this factor. Skip to the next factor
  if nextFactor is not None:
    newWidth: int = imageWidth // nextFactor
    newHeight: int = imageHeight // nextFactor
    directory: str = os.path.join(
        os.getcwd(),
        'images',
        'processed',
        f'{newHeight}x{newWidth}')

    if os.path.exists(directory):
      return True
  return False


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
  image = ColourFiltering.toGreyscale(image)

  # convert to binary
  image = ColourFiltering.toBinary(image, cutoff)

  # remove background
  image = ColourFiltering.removeBackground(
      image, f'{imageDirectory}/{bb_filename}')

  # convert to binary
  image = ColourFiltering.toBinary(image, cutoff)

  # apply opening
  image = MorphologicalFiltering.opening(image, kernelSize)

  # apply closing
  image = MorphologicalFiltering.closing(image, kernelSize)

  # downsample
  image = ColourFiltering.downSample(image, downsampleFactor)

  return image


if __name__ == '__main__':
  preprocessAllImages()
