import matplotlib.pyplot as plt

class ColourFiltering:
    """
    A class to perform colour filtering on images
    """

    @staticmethod
    def toGreyscale(image):
        """
        Convert the coloured image to a greyscale image
        
        :param image: the image to convert (numpy array)
        
        :return: the greyscale image (numpy array)
        """

        return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

    @staticmethod
    def toBinary(image, threshold):
        """
        Convert the greyscale image to a binary image
        
        :param image: the image to convert (numpy array)
        :param threshold: the threshold to use for the conversion (float)

        :return: the binary image (numpy array)
        """
        
        return image > threshold

    @staticmethod
    def removeBackground(image):
        """
        Remove the background from the image using the reference image
        
        :param image: the image to remove the background from (numpy array)

        :return: the image with the background removed (numpy array)
        """
        # Get the reference image
        ref_image_filepath = "./images/snapshot_618423.png"
        ref_image = plt.imread(ref_image_filepath)

        # Convert the reference image to greyscale
        ref_image = ColourFiltering.toGreyscale(ref_image)

        # Convert the reference image to binary
        ref_image = ColourFiltering.toBinary(ref_image, 0.5)

        # invert the reference image
        ref_image = ~ref_image

        # Perform a elementwise OR operation on the image and the reference image
        return image | ref_image