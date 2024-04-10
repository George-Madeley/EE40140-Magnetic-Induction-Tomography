import math

import numpy as np


class FrequencyFiltering():
  """
  Class for applying linear filters to an image
  """

  @staticmethod
  def applyFilter(
          image: np.ndarray,
          filter_name: str,
          kernel_size: int,
          **kwargs) -> np.ndarray:
    """
    Applies a linear filter to an image

    :param image: The image to be filtered
    :param filter_name: The name of the filter. Possible values are
        - 'gaussian',
        - 'box',
        - 'butterworth_low_pass',
        - 'low_pass'
    :param kernel_size: The size of the kernel
    :param kwargs: The arguments for the filter. Possible values are:
        - 'order': The order of the filter. Required for
                   'butterworth_low_pass' and 'contra_harmonic_mean'
        - 'cutoff': The cutoff frequency. Required for
                   'butterworth_low_pass' and 'low_pass'
        - 'padding': The type of padding to use. Possible values are:
            - 'constant'
            - 'edge'
            - 'linear_ramp'

    :return: The filtered image
    """

    # Get the padding type to be used for the convolution
    padding = kwargs.get('padding', 'constant')

    # Some filters have kernels and can be applied using the frequency
    # domain algorithm. Other filters do not have kernels and can only be
    # applied using the spatial domain algorithm. Check if the filter has a
    # kernel and if so, apply it using the frequency domain algorithm.
    # Otherwise, apply it using the spatial domain algorithm.
    if filter_name == 'gaussian':
      # Get the Gaussian kernel of the specified size and apply it using
      # the frequency domain algorithm.
      kernel = FrequencyFiltering.getGaussianKernel(kernel_size)
      return FrequencyFiltering.calculateFrequencyDomainConvolution(
          image, kernel, padding)
    elif filter_name == 'box':
      # Get the box kernel of the specified size and apply it using the
      # frequency domain algorithm.
      kernel = FrequencyFiltering.getBoxKernel(kernel_size)
      return FrequencyFiltering.calculateFrequencyDomainConvolution(
          image, kernel, padding)
    elif filter_name == 'butterworth_low_pass':
      # Get the order and cutoff frequency from the kwargs to be used in
      # the Butterworth low pass filter.
      order = kwargs.get('order', 2)
      cutoff = kwargs.get('cutoff', 50.0)
      # Get the Butterworth low pass filter of the specified size and
      # apply it using the frequency domain algorithm.
      kernel = FrequencyFiltering.getButterworthLowPassFilter(
          kernel_size, cutoff, order)
      return FrequencyFiltering.calculateFrequencyDomainConvolution(
          image, kernel, padding)
    elif filter_name == 'low_pass':
      # Get the cutoff frequency from the kwargs to be used in the low
      # pass filter.
      cutoff = kwargs.get('cutoff', 50.0)
      # Get the low pass filter of the specified size and apply it using
      # the frequency domain algorithm.
      kernel = FrequencyFiltering.getLowPassFilter(kernel_size, cutoff)
      return FrequencyFiltering.calculateFrequencyDomainConvolution(
          image, kernel, padding)
    else:
      # If the filter name is not recognized, raise an error.
      raise Exception('Invalid filter name.')

  @staticmethod
  def calculateFrequencyDomainConvolution(
          image: np.ndarray,
          kernel: np.ndarray,
          padding: str = 'constant') -> np.ndarray:
    """
    Performs a convolution on an image using a kernel using the Fast Fourier
    Transform algorithm.

    :param image: The image to be convolved
    :param kernel: The kernel to convolve the image with
    :param padding: The type of padding to use

    :return: The convolved image
    """

    # Creates tuple for size of padded image and kernel
    new_size = (
        image.shape[0] +
        kernel.shape[0] -
        1,
        image.shape[1] +
        kernel.shape[1] -
        1)
    convolved_image = np.zeros(shape=new_size, dtype=image.dtype)

    # Calculates half the size of the image and kernel respectfully in both
    # dimensions
    half_image = ((image.shape[0] - 1) / 2, (image.shape[1] - 1) / 2)
    half_kernal = ((kernel.shape[0] - 1) / 2, (kernel.shape[1] - 1) / 2)

    # Pads the image with duplicate values and the kernel with 0s
    pad_image = np.pad(image, pad_width=(
        (math.floor(half_kernal[0]), math.ceil(half_kernal[0])),
        (math.floor(half_kernal[1]), math.ceil(half_kernal[1]))
    ), mode=padding)

    pad_kernel = np.zeros(shape=new_size)
    pad_kernel[0: kernel.shape[0], 0: kernel.shape[1]] = kernel

    # Calculates the Fourier transforms for the image and kernel
    fft_image = np.fft.fft2(pad_image)
    fft_kernel = np.fft.fft2(pad_kernel)

    # Performs the convolutions, inverses the fourier transforms and
    # extracts the real part of each element
    convolved_image = np.real(np.fft.ifft2(fft_image * fft_kernel))

    # Function to calculate the padding of the convoluted image
    def bounds(axis): return kernel.shape[axis] - 1

    # Removes the padding from the convoluted image
    convolved_image = convolved_image[bounds(
        0): new_size[0], bounds(1): new_size[1]]

    return convolved_image
