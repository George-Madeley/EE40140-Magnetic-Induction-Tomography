from skimage.morphology import erosion, dilation, opening, closing, square
import numpy as np

def erosion(image: np.ndarray, size: int = 3) -> np.ndarray:
  """
  Apply an erosion operation to the image

  :param image: the image to apply the operation to (numpy array)
  :param size: the size of the kernel (int)

  :return: the eroded image (numpy array)
  """
  return erosion(image.astype(bool), selem=square(size)).astype(image.dtype)


def dilation(image: np.ndarray, size: int = 3) -> np.ndarray:
  """
  Apply a dilation operation to the image

  :param image: the image to apply the operation to (numpy array)
  :param size: the size of the kernel (int)

  :return: the dilated image (numpy array)
  """
  return dilation(image.astype(bool), selem=square(size)).astype(image.dtype)


def opening(image: np.ndarray, size: int = 3) -> np.ndarray:
  """
  Apply an opening operation to the image

  :param image: the image to apply the operation to (numpy array)
  :param size: the size of the kernel (int)

  :return: the opened image (numpy array)
  """
  return opening(image.astype(bool), square(size)).astype(image.dtype)


def closing(image: np.ndarray, size: int = 3) -> np.ndarray:
  """
  Apply a closing operation to the image

  :param image: the image to apply the operation to (numpy array)
  :param size: the size of the kernel (int)

  :return: the closed image (numpy array)
  """
  return closing(image.astype(bool), square(size)).astype(image.dtype)
