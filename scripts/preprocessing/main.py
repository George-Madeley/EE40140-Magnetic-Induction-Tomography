import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colourFiltering import ColourFiltering
from frequencyFiltering import FrequencyFiltering
from morphologicalFiltering import MorphologicalFiltering

def main():
    """
    Overlaps multiple images from the 'images' directory and displays the result.
    Each image is converted to grayscale and then overlapped using alpha blending.
    The resulting overlapped image is displayed using matplotlib.
    """
    # Get the image directory
    imageDirectory = "./images"

    # load data from csv with headers
    df = pd.read_csv('./input_data.csv')

    while True:
        # gerenate a random number between 0 and the number of records
        index = np.random.randint(0, len(df))

        # get the filename of a random record
        imageFilename = df.iloc[index]['filepath']
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

        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')

        plt.show()


    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    plt.show()

main()