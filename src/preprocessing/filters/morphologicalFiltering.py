import skimage
import numpy as np

def erosion(image: np.ndarray, size: int = 3) -> np.ndarray:
  """
  Apply an erosion operation to the image

  :param image: the image to apply the operation to (numpy array)
  :param size: the size of the kernel (int)

  :return: the eroded image (numpy array)
  """
  kernel = np.ones((size, size))
  return skimage.morphology.erosion(image.astype(bool), kernel).astype(image.dtype)


def dilation(image: np.ndarray, size: int = 3) -> np.ndarray:
  """
  Apply a dilation operation to the image

  :param image: the image to apply the operation to (numpy array)
  :param size: the size of the kernel (int)

  :return: the dilated image (numpy array)
  """
  kernel = np.ones((size, size))
  return skimage.morphology.dilation(image.astype(bool), kernel).astype(image.dtype)


def opening(image: np.ndarray, size: int = 3) -> np.ndarray:
  """
  Apply an opening operation to the image

  :param image: the image to apply the operation to (numpy array)
  :param size: the size of the kernel (int)

  :return: the opened image (numpy array)
  """
  kernel = np.ones((size, size))
  return skimage.morphology.opening(image.astype(bool), kernel).astype(image.dtype)


def closing(image: np.ndarray, size: int = 3) -> np.ndarray:
  """
  Apply a closing operation to the image

  :param image: the image to apply the operation to (numpy array)
  :param size: the size of the kernel (int)

  :return: the closed image (numpy array)
  """
  kernel = np.ones((size, size))
  return skimage.morphology.closing(image.astype(bool), kernel).astype(image.dtype)
