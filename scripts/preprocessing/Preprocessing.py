import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colourFiltering import ColourFiltering
from frequencyFiltering import FrequencyFiltering
from morphologicalFiltering import MorphologicalFiltering

def downSample(image, factor):
    downsampled_image = image[::factor, ::factor]
    return downsampled_image

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

    # downsample
    image = downSample(image, 8)

    return image

def preprocessAllImages():
    # load in input_data.csv
    df = pd.read_csv('input_data.csv')

    for i, row in df.iterrows():
        # Get the filename
        imageFilename = row['filepath']

        print(f'Processing image {imageFilename}')

        # Preprocess the image
        image = Preprocess(imageFilename)

        # get file name without extension
        imageFilename = os.path.splitext(imageFilename)[0]

        # save image
        savePath = os.path.join('./images(60x80)', f'{imageFilename}.npy')
        np.save(savePath, image)

if __name__ == '__main__':
    preprocessAllImages()