import numpy as np
import matplotlib.pyplot as plt

class ColourFiltering:
  """
  A class to perform colour filtering on images
  """

  @staticmethod
  def toGreyscale(image: np.ndarray) -> np.ndarray:
    """
    Convert the coloured image to a greyscale image

    :param image: the image to convert (numpy array)

    :return: the greyscale image (numpy array)
    """
    return 0.2126 * image[:, :, 0] + 0.7152 * \
        image[:, :, 1] + 0.0722 * image[:, :, 2]

  @staticmethod
  def toBinary(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert the greyscale image to a binary image

    :param image: the image to convert (numpy array)
    :param threshold: the threshold to use for the conversion (float)

    :return: the binary image (numpy array)
    """
    return image > threshold

  @staticmethod
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
    ref_image = ColourFiltering.toGreyscale(ref_image)

    # Convert the reference image to binary
    ref_image = ColourFiltering.toBinary(ref_image, 0.5)

    # invert the reference image
    ref_image = ~ref_image

    # Set the left and right sides of the image to 1
    ref_image[:, :50] = 1
    ref_image[:, -100:] = 1

    # Perform an elementwise OR operation on the image and the reference image
    return np.array(image, dtype=bool) | np.array(ref_image, dtype=bool)

  @staticmethod
  def downSample(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsamples an image by a given factor.

    :param image: The input image to be downsampled (numpy array).
    :param factor: The downsampling factor (int).

    :return: The downsampled image (numpy array).
    """
    downsampled_image = image[::factor, ::factor]
    return downsampled_image
