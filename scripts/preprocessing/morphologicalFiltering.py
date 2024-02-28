from skimage.morphology import erosion, dilation, opening, closing, square

class MorphologicalFiltering:
    """
    A class that contains methods for morphological filtering
    """

    @staticmethod
    def erosion(image, size=3):
        """
        Apply an erosion operation to the image
        
        :param image: the image to apply the operation to (numpy array)
        :param size: the size of the kernel (int)
        
        :return: the eroded image (numpy array)
        """
        return erosion(image, selem=square(size))

    @staticmethod
    def dilation(image, size=3):
        """
        Apply a dilation operation to the image
        
        :param image: the image to apply the operation to (numpy array)
        :param size: the size of the kernel (int)
        
        :return: the dilated image (numpy array)
        """
        return dilation(image, selem=square(size))

    @staticmethod
    def opening(image, size=3):
        """
        Apply an opening operation to the image
        
        :param image: the image to apply the operation to (numpy array)
        :param size: the size of the kernel (int)
        
        :return: the opened image (numpy array)
        """
        return opening(image, square(size))

    @staticmethod
    def closing(image, size=3):
        """
        Apply a closing operation to the image
        
        :param image: the image to apply the operation to (numpy array)
        :param size: the size of the kernel (int)
        
        :return: the closed image (numpy array)
        """
        return closing(image, square(size))