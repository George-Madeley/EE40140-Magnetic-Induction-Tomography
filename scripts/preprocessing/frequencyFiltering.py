import numpy as np
import math


class FrequencyFiltering():
    """
    Class for applying linear filters to an image
    """

    @staticmethod
    def applyFilter(image, filter_name, kernel_size, **kwargs):
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
    def calculateFrequencyDomainConvolution(image, kernel, padding='constant'):
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

    @staticmethod
    def getGaussianKernel(size):
        """
        Creates a Gaussian kernel of size (size x size) with standard deviation
        sigma

        :param size: The size of the kernel

        :return: The Gaussian kernel
        """
        # Calculates the standard deviation
        stdiv = (size - 1) / 6

        # Creates a kernel of zeros
        kernel = np.zeros(shape=(size, size))
        # Calculates the center of the kernel
        center = (size - 1) / 2
        # Creates a vector of values from -center to center
        vector = np.linspace(-center, center, size)
        vector = vector ** 2
        # Create a matrix of distances from the center
        distances = np.sqrt(np.add.outer(vector, vector))
        # Calculates the constant for the Gaussian function
        constant = 1 / (2 * math.pi * stdiv ** 2)
        # Calculates the Gaussian filter
        kernel = constant * np.exp(-0.5 * distances ** 2 / stdiv ** 2)
        # Normalizes the kernel
        kernel /= np.sum(kernel)
        return kernel

    @staticmethod
    def getBoxKernel(size):
        """
        Creates a box kernel of size (size x size)

        :param size: The size of the kernel

        :return: The box kernel
        """
        # Creates a kernel of ones
        kernel = np.ones(shape=(size, size))
        # Normalizes the kernel
        kernel /= np.sum(kernel)
        return kernel

    @staticmethod
    def getButterworthLowPassFilter(size, cutoff, order):
        """
        Creates a Butterworth low pass filter of size (size x size) with cutoff
        frequency cutoff mand order order

        :param size: The size of the filter
        :param cutoff: The cutoff frequency
        :param order: The order of the filter

        :return: The Butterworth low pass filter
        """
        # Creates a kernel of zeros
        kernel = np.zeros(shape=(size, size))
        # Calculates the center of the kernel
        center = (size - 1) / 2
        # Creates a vector of values from -center to center
        vector = np.linspace(-center, center, size)
        vector = vector ** 2
        # Create a matrix of distances from the center
        distances = np.sqrt(np.add.outer(vector, vector))
        # Calculates the Butterworth low pass filter
        kernel = 1 / (1 + (distances / cutoff) ** (2 * order))
        # Normalizes the kernel
        kernel /= np.sum(kernel)
        return kernel

    @staticmethod
    def getLowPassFilter(size, cutoff):
        """
        Creates a low pass filter of size (size x size) with cutoff frequency
        cutoff

        :param size: The size of the filter
        :param cutoff: The cutoff frequency

        :return: The low pass filter
        """
        # Creates a kernel of zeros
        kernel = np.zeros(shape=(size, size))
        # Calculates the center of the kernel
        center = (size - 1) / 2
        # Creates a vector of values from -center to center
        vector = np.linspace(-center, center, size)
        # Calculates the low pass filter
        low_pass_filter = np.where(np.abs(vector) <= cutoff, 1, 0)
        # Creates the low pass filter
        kernel = np.outer(low_pass_filter, low_pass_filter)
        return kernel
