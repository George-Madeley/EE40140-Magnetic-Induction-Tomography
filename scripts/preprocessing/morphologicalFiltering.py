import numpy as np

def erosion(image, size = 3):
    """
    Apply an erosion operation to the image
    
    :param image: the image to apply the operation to (numpy array)
    :param size: the size of the kernel (int)
    
    :return: the eroded image (numpy array)
    """

    # Create a kernel of ones
    kernel = np.ones((size, size))

    # Get the size of the image and the kernel
    image_size = image.shape
    kernel_size = kernel.shape

    # Get the size of the padding required
    padding = (kernel_size[0] - 1, kernel_size[1] - 1)

    # Pad the image
    padded_image = np.pad(image, padding, mode='constant')

    # Create a new image to store the result
    result = np.zeros(image_size)

    # Iterate over the image
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            # Get the region of the image that corresponds to the kernel
            region = padded_image[i:i + kernel_size[0], j:j + kernel_size[1]]

            # Perform the erosion operation
            result[i, j] = np.min(region * kernel)

    return result

def dilation(image, size = 3):
    """
    Apply a dilation operation to the image
    
    :param image: the image to apply the operation to (numpy array)
    :param size: the size of the kernel (int)

    :return: the dilated image (numpy array)
    """

    # Create a kernel of ones
    kernel = np.ones((size, size))

    # Get the size of the image and the kernel
    image_size = image.shape
    kernel_size = kernel.shape

    # Get the size of the padding required
    padding = (kernel_size[0] - 1, kernel_size[1] - 1)

    # Pad the image
    padded_image = np.pad(image, padding, mode='constant')

    # Create a new image to store the result
    result = np.zeros(image_size)

    # Iterate over the image
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            # Get the region of the image that corresponds to the kernel
            region = padded_image[i:i + kernel_size[0], j:j + kernel_size[1]]

            # Perform the dilation operation
            result[i, j] = np.max(region * kernel)

    return result

def opening(image, size = 3):
    """
    Apply an opening operation to the image
    
    :param image: the image to apply the operation to (numpy array)
    :param size: the size of the kernel (int)

    :return: the opened image (numpy array)
    """
    
    # Perform an erosion operation
    eroded = erosion(image, size)

    # Perform a dilation operation
    dilated = dilation(eroded, size)

    return dilated

def closing(image, size = 3):
    """
    Apply a closing operation to the image
    
    :param image: the image to apply the operation to (numpy array)
    :param size: the size of the kernel (int)

    :return: the closed image (numpy array)
    """

    # Perform a dilation operation
    dilated = dilation(image, size)

    # Perform an erosion operation
    eroded = erosion(dilated, size)

    return eroded