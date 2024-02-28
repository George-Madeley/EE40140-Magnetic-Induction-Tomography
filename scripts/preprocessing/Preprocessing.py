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

def Preprocess(imageFilename, kernelSize=5, cutoff=0.5, downsampleFactor=1):
    imageDirectory = './images/original'

    # get the filename
    imagePath = os.path.join(imageDirectory, imageFilename)

    # get image from file
    image = np.array(plt.imread(imagePath))

    # convert to greyscale
    image = ColourFiltering.toGreyscale(image)

    # convert to binary
    image = ColourFiltering.toBinary(image, cutoff)

    # remove background
    image = ColourFiltering.removeBackground(image)

    # apply low pass filter
    image = FrequencyFiltering.applyFilter(image, 'low_pass', kernelSize)

    # convert to binary
    image = ColourFiltering.toBinary(image, cutoff)

    # apply closing
    image = MorphologicalFiltering.closing(image, kernelSize)

    # downsample
    image = downSample(image, downsampleFactor)

    return image

def preprocessAllImages():
    # load in input_data.csv
    df = pd.read_csv('input_data.csv')

    for i, row in df.iterrows():
        # Get the filename
        imageFilename = row['filepath']

        print(f'Processing image {imageFilename}')

        # Preprocess the image
        image = Preprocess(imageFilename, downsampleFactor=downSampleFactor)

        # save image
        savePath = os.path.join('./images(60x80)', f'{imageFilename}.npy')
        np.save(savePath, image)

if __name__ == '__main__':
    preprocessAllImages()