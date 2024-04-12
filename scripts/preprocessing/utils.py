import os
from typing import List


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
  imageHeight: int
) -> bool:
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