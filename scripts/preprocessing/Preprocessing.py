import os
import matplotlib.pyplot as plt
import numpy as np

from colourFiltering import ColourFiltering
from frequencyFiltering import FrequencyFiltering
from morphologicalFiltering import MorphologicalFiltering
from cleanUp import cleanUp

def preprocessAllImages():
    # load in the dataset
    df = cleanUp.main()

    imageWidth = 640
    imageHeight = 480
    commonFactors = getCommonFactors(imageWidth, imageHeight)

    for idx in range(len(commonFactors)):
        factor = commonFactors[idx]
        nextFactor = commonFactors[idx + 1] if idx + 1 < len(commonFactors) else None

        if checkForProcessedFactors(nextFactor, imageWidth, imageHeight):
            continue

        print(f'Processing images with factor {factor}')

        newWidth = imageWidth // factor
        newHeight = imageHeight // factor

        directory = os.path.join(os.getcwd(), 'images', 'processed', f'{newHeight}x{newWidth}')
        os.makedirs(directory, exist_ok=True)

        for i, row in df.iterrows():
            # Get the filename
            bb_filename = row['bb_filename']
            cc_filename = row['cc_filename']
            sample = row['sample']

            savePath = os.path.join(directory, cc_filename)

            if os.path.exists(savePath):
                continue

            print(f'Processing image {cc_filename} to {newHeight}x{newWidth} sample: {sample}, i: {i}')

            # Preprocess the image
            image = Preprocess(bb_filename, cc_filename, downsampleFactor=factor)

            # save image
            plt.imsave(savePath, image, cmap='gray')

def getCommonFactors(a, b, maxFactor=32):
    factors = []
    for i in range(1, min(a, b) + 1):
        if i > maxFactor:
            break

        if a % i == 0 and b % i == 0:
            factors.append(i)
    return factors

def checkForProcessedFactors(nextFactor, imageWidth, imageHeight):
    # If the next factor directory exists, then the images have already been
    # processed for this factor. Skip to the next factor
    if nextFactor is not None:
        newWidth = imageWidth // nextFactor
        newHeight = imageHeight // nextFactor
        directory = os.path.join(os.getcwd(), 'images', 'processed', f'{newHeight}x{newWidth}')

        if os.path.exists(directory):
            return True
    return False

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
    image = ColourFiltering.downSample(image, downsampleFactor)

    return image

if __name__ == '__main__':
    preprocessAllImages()