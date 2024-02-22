import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plotVoltageDifferenceHeatMap(index: int = 0):
    # load data from csv with headers
    df = pd.read_csv('./input_data.csv')

    # get the names of the sensor columns
    label_names = df.filter(regex='sensor_\d+').columns

    # get the record at the given index
    values = df.iloc[index][label_names]

    # Create matrix 16 x 16 filledwith zeros
    matrix = np.zeros((16, 16))

    # Store the values in the upper triangle of the matrix
    matrix[np.triu_indices(16, 1)] = values

    # Set the lower triangle of the matrix to the transpose of the upper triangle
    matrix += matrix.T

    # plot the matrix as a heatmap
    sns.heatmap(matrix, cmap='coolwarm', annot=False)
    plt.show()

plotVoltageDifferenceHeatMap(-1)