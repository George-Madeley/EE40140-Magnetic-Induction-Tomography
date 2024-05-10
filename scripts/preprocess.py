import os

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import cleanUp
from preprocessing import formatting
from preprocessing.filters import colourFiltering, morphologicalFiltering
from preprocessing.utils import getCommonFactors, getImage


def preprocessAllImages(
    toImagePreprocess: bool = False
) -> None:
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
  cleanUp.deleteImages()

  df = cleanUp.cleanUp()

  if toImagePreprocess:
    print("Preprocessing Images...")
    for i, row in df.iterrows():
      # Get the filename
      bb_filename: str = row['bb_filename']
      cc_filename: str = row['cc_filename']
      cutoff: float = row['cutoff']
      sample: str = row['sample']

      imageWidth: int = 640
      imageHeight: int = 480
      commonFactors: List[int] = getCommonFactors(imageWidth, imageHeight)

      # Preprocess the image
      PreprocessImage( bb_filename, cc_filename, sample, commonFactors, cutoff=cutoff)

  df = formatting.normalise(df)

  material_dfs = formatting.splitDataframeMaterial(df)

  for material_tuple in material_dfs:
    material, material_df = material_tuple
    feature = 'shape' if material == 'iron' else 'sample'

    if material != 'unknown':
      material_df = formatting.getEvenDistribution(
        material_df, distFeature=feature)
      material_df = formatting.splitDataframeFeature(material_df, feature)
    else:
      # shuffle the unknown samples
      material_df = material_df.sample(frac=1).reset_index(drop=True)
    
    material_df = formatting.oneHotEncode(material_df, feature)
    material_df.to_csv(f'./data/data_samples_{material}.csv', index=False)

  print('Preprocessing complete.')


def PreprocessImage(
  bb_filename: str,
  cc_filename: str,
  sample: str,
  commonFactors: list[int],
  kernelSize: int = 5,
  cutoff: float = 0.5,
  imageWidth: int = 640,
  imageHeight: int = 480
) -> None:
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
      None
  """

  isImageProcessed: bool = True
  for factor in commonFactors:
    newWidth: int = imageWidth // factor
    newHeight: int = imageHeight // factor
    directory: str = os.path.join(
        'images',
        'processed',
        f'{newHeight}x{newWidth}')
    os.makedirs(directory, exist_ok=True)
    savePath: str = os.path.join(directory, cc_filename)
    if not os.path.exists(savePath):
      isImageProcessed = False
      break

  if isImageProcessed:
    return

  image = getImage(cc_filename, commonFactors, imageWidth, imageHeight)
  if image is None: return


  if sample == '0':
    # Make all null images white.
    image = np.ones((imageHeight, imageWidth, 3))
  else:
    image = colourFiltering.toGreyscale(image)
    image = colourFiltering.toBinary(image, cutoff)
    image = colourFiltering.removeBackground(
        image, os.path.join('images', 'original', bb_filename))
    image = colourFiltering.toBinary(image, cutoff)
    image = morphologicalFiltering.opening(image, kernelSize)
    image = morphologicalFiltering.closing(image, kernelSize)
  
  
  for factor in commonFactors:
    newWidth: int = imageWidth // factor
    newHeight: int = imageHeight // factor
    directory: str = os.path.join(
        'images',
        'processed',
        f'{newHeight}x{newWidth}')
    os.makedirs(directory, exist_ok=True)
    savePath: str = os.path.join(directory, cc_filename)

    # downsample
    downSampledImage = colourFiltering.downSample(image, factor)

    # save image
    plt.imsave(savePath, downSampledImage, cmap='gray')


  print(f"Processed image {cc_filename}\tSample: {sample}\tcutoff:{cutoff}")
  return True


if __name__ == '__main__':
  preprocessAllImages(
    toImagePreprocess=True
  )



