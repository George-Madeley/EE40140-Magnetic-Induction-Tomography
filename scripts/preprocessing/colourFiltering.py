import matplotlib.pyplot as plt

def toGreyscale(image):
    """
    Convert the coloured image to a greyscale image
    
    :param image: the image to convert (numpy array)
    
    :return: the greyscale image (numpy array)
    """

    return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

def toBinary(image, threshold):
    """
    Convert the greyscale image to a binary image
    
    :param image: the image to convert (numpy array)
    :param threshold: the threshold to use for the conversion (float)

    :return: the binary image (numpy array)
    """
    
    return image > threshold