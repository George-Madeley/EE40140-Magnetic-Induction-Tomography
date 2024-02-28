import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colourFiltering import toGreyscale

def main():
    """
    Overlaps multiple images from the 'images' directory and displays the result.
    Each image is converted to grayscale and then overlapped using alpha blending.
    The resulting overlapped image is displayed using matplotlib.
    """
    # Get the image directory
    image_dir = "./images"

    # load data from csv with headers
    df = pd.read_csv('./input_data.csv')

    while True:
        # gerenate a random number between 0 and the number of records
        index = np.random.randint(0, len(df))

        # get the filename of a random record
        image_file = df.iloc[index]['filepath']
        image_path = os.path.join(image_dir, image_file)

        # get image from file
        rgb_img = np.array(plt.imread(image_path))

        # convert to greyscale
        bw_img = toGreyscale(rgb_img)

        fig, ax = plt.subplots()
        ax.imshow(bw_img, cmap='gray')

        plt.show()


    fig, ax = plt.subplots()
    ax.imshow(bw_img, cmap='gray')

    plt.show()

main()