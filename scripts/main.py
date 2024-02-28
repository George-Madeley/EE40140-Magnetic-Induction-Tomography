import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # load in input_data.csv
    df = pd.read_csv('input_data.csv')

    column_names = df.filter(regex='sensor_\d+').columns
    voltages = df[column_names].values

    outputImages = np.zeros((len(df), 60, 80))

    for i, row in df.iterrows():
        # Get the filename
        imageFilename = row['filepath']
        
        imageFilePath = os.path.join('./images/processed/60x80', imageFilename)

        # Read the .png file
        image = plt.imread(imageFilePath)

        # Average the four channels
        image = np.mean(image, axis=2)

        # Add the image to the outputImages array
        outputImages[i] = image





if __name__ == '__main__':
    main()

    
