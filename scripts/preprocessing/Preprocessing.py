import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colourFiltering import ColourFiltering
from frequencyFiltering import FrequencyFiltering
from morphologicalFiltering import MorphologicalFiltering
from cleanUp import cleanUp

def downSample(image, factor):
    downsampled_image = image[::factor, ::factor]
    return downsampled_image

def Preprocess(bb_filename, cc_filename, kernelSize=5, cutoff=0.5, downsampleFactor=1):
    imageDirectory = './images/original'

    # get the filename
    imagePath = os.path.join(imageDirectory, cc_filename)

    # get image from file
    image = np.array(plt.imread(imagePath))

    # convert to greyscale
    image = ColourFiltering.toGreyscale(image)

    # convert to binary
    image = ColourFiltering.toBinary(image, cutoff)

    # remove background
    image = ColourFiltering.removeBackground(image, f'{imageDirectory}/{bb_filename}')

    # convert to binary
    image = ColourFiltering.toBinary(image, cutoff)

    # apply opening
    image = MorphologicalFiltering.opening(image, kernelSize)

    # apply closing
    image = MorphologicalFiltering.closing(image, kernelSize)

    # downsample
    image = downSample(image, downsampleFactor)

    return image

def preprocessAllImages():
    # load in the dataset
    df = cleanUp.main

    downSampleFactor = 8
    newWidth = 640 // downSampleFactor
    newHeight = 480 // downSampleFactor

    directory = f'./images/processed/{newHeight}x{newWidth}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, row in df.iterrows():
        # Get the filename
        bb_filename = row['bb_filename']
        cc_filename = row['cc_filename']

        print(f'Processing image {cc_filename}')

        # Preprocess the image
        image = Preprocess(bb_filename, cc_filename, downsampleFactor=downSampleFactor)

        # save image
        savePath = os.path.join(directory, cc_filename)
        plt.imsave(savePath, image, cmap='gray')

if __name__ == '__main__':
    preprocessAllImages()