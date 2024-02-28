import os
import matplotlib.pyplot as plt
import numpy as np

from colourFiltering import ColourFiltering
from frequencyFiltering import FrequencyFiltering
from morphologicalFiltering import MorphologicalFiltering

def Preprocess(imageFilename):
    imageDirectory = './images'

    # get the filename
    imagePath = os.path.join(imageDirectory, imageFilename)

    # get image from file
    image = np.array(plt.imread(imagePath))

    # convert to greyscale
    image = ColourFiltering.toGreyscale(image)

    # convert to binary
    image = ColourFiltering.toBinary(image, 0.5)

    # remove background
    image = ColourFiltering.removeBackground(image)

    # apply low pass filter
    image = FrequencyFiltering.applyFilter(image, 'low_pass', 5, cutoff=50.0)

    # convert to binary
    image = ColourFiltering.toBinary(image, 0.5)

    # apply closing
    image = MorphologicalFiltering.closing(image, 5)
