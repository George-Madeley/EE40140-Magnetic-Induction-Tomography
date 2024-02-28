import numpy as np
import math

def lowPassFilter(image, size = 3, padding = 'constant'):
    """
    Apply a low pass filter to the image
    
    :param image: the image to apply the filter to (numpy array)
    :param size: the size of the filter (int)
    
    :return: the filtered image (numpy array)
    """

    # Creates a kernel of ones
    kernel = np.ones(shape=(size, size))
    # Normalizes the kernel
    kernel /= np.sum(kernel)

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