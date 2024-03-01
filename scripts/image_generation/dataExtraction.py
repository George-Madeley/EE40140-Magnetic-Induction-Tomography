import pandas as pd
import numpy as np
import os
from typing import Tuple
from pandas import DataFrame
import torch

import matplotlib.pyplot as plt

def dataExtraction(
        downScaleFactor: int = 1, 
        approximations: Tuple[float, float] = (0.01, 0.99),
        batchSize: int = 50
    ) -> torch.utils.data.DataLoader:
    """
    Extracts data from a CSV file and prepares it for training a neural network.

    Args:
        downScaleFactor (int): The factor by which the images should be
        downscaled. Default is 1.
        approximations (Tuple[float, float]): The lower and upper bounds for
        approximating the images. Default is (0.01, 0.99).
        batchSize (int): The batch size for the data loader. Default is 50.

    Returns:
        torch.utils.data.DataLoader: A data loader object containing the
        training data.

    """
    # load in input_data.csv
    df = pd.read_csv('data.csv')

    # Get the background voltage and the sensor voltage columns
    voltages = getVoltages(df)
    images = getImages(df, approximations, downScaleFactor)

    trainSet = torch.utils.data.TensorDataset(images, voltages)

    trainLoader = torch.utils.data.DataLoader(
        trainSet, batch_size=batchSize, shuffle=True
    )

    return trainLoader

def getImages(
    df: DataFrame,
    approximations: Tuple[float, float],
    downScaleFactor: int
    ) -> torch.Tensor:
    """
    Extracts and processes images from a dataframe.

    :param df: The dataframe containing image information.
    :param approximations: A tuple of two values representing the approximations
    for zero and one values.
    :param downScaleFactor: The factor by which the original image dimensions
    are scaled down.

    :return: A tensor of processed images.

    :raises: None
    """

    # Define the original width and height
    originalWidth = 640
    originalHeight = 480

    # Define the new width and height
    newWidth = originalWidth // downScaleFactor
    newHeight = originalHeight // downScaleFactor


    outputImages = torch.zeros((len(df), 1, 60, 80))

    for i, row in df.iterrows():
        # Get the filename
        imageFilename = row['cc_filename']
        
        imageFilePath = os.path.join(f'./images/processed/{newHeight}x{newWidth}', imageFilename)

        # Read the .png file
        image = plt.imread(imageFilePath)
        # Average the first three channels
        image = np.mean(image[:, :, :3], axis=2)

        # Find all the zero values and replace them with 0.01
        zero_values = image == 0
        image[zero_values] = approximations[0]

        # Find all the one values nd replace them with 0.99
        one_values = image == 1
        image[one_values] = approximations[1]

        # Convert the image to a PyTorch tensor
        image_tensor = torch.from_numpy(image).float()

        # Reshape the tensor to the expected input shape
        image_tensor = image_tensor.view(1, newHeight, newWidth)

        # Add the image to the outputImages array
        outputImages[i] = image_tensor

    return outputImages

def getVoltages(df: DataFrame) -> torch.Tensor:
    """
    Extracts voltage data from a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame containing voltage data.

    Returns:
        torch.Tensor: A tensor of normalized voltage values.

    """
    bb_names = df.filter(regex='bb_\d+').columns
    cc_names = df.filter(regex='cc_\d+').columns
    # combine the two lists and return the result
    column_names = bb_names.append(cc_names)

    voltages = torch.tensor(df[column_names].values)

    # Normalize the voltages by dividing by 2e4
    voltages = voltages / 2e4

    return voltages
