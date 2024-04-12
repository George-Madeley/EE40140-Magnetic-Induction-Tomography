import numpy as np
import matplotlib.pyplot as plt

def toGreyscale(image: np.ndarray) -> np.ndarray:
  """
  Convert the coloured image to a greyscale image

  :param image: the image to convert (numpy array)

  :return: the greyscale image (numpy array)
  """
  r_ch = image[:, :, 0]
  g_ch = image[:, :, 1]
  b_ch = image[:, :, 2]
  return 0.2126 * r_ch + 0.7152 * g_ch + 0.0722 * b_ch


def toBinary(image: np.ndarray, threshold: float) -> np.ndarray:
  """
  Convert the greyscale image to a binary image

  :param image: the image to convert (numpy array)
  :param threshold: the threshold to use for the conversion (float)

  :return: the binary image (numpy array)
  """

  # If the pixel value is greater than the threshold, set it to 1, otherwise set it to 0
  newImage = np.where(image > threshold, 1, 0)

  # Change the data type of newImage to np.uint8
  newImage = newImage.astype(np.uint8)

  return newImage


def removeBackground(
        image: np.ndarray,
        ref_image_filepath: str) -> np.ndarray:
  """
  Remove the background from the image using the reference image

  :param image: the image to remove the background from (numpy array)
  :param ref_image_filepath: the filepath of the reference image (str)

  :return: the image with the background removed (numpy array)
  """
  # Get the reference image
  ref_image = plt.imread(ref_image_filepath)

  # Convert the reference image to greyscale
  ref_image = toGreyscale(ref_image)

  # Convert the reference image to binary
  ref_image = toBinary(ref_image, 0.5)

  # Set all the pixels that are 1 to 0 and vice versa
  ref_image = 1 - ref_image

  # Set the left and right sides of the image to 1
  ref_image[:, :50] = 1
  ref_image[:, -100:] = 1

  # Perform an elementwise OR operation on the image and the reference image
  return np.array(image, dtype=bool) | np.array(ref_image, dtype=bool)


def downSample(image: np.ndarray, factor: int) -> np.ndarray:
  """
  Downsamples an image by a given factor.

  :param image: The input image to be downsampled (numpy array).
  :param factor: The downsampling factor (int).

  :return: The downsampled image (numpy array).
  """
  downsampled_image = image[::factor, ::factor]
  return downsampled_image
