import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plotVoltageDifferenceLineGraph(index: int = 0):
    # load data from csv with headers
    df = pd.read_csv('./input_data.csv')

    # get the names of the sensor columns
    label_names = df.filter(regex='sensor_\d+').columns
    values = df.iloc[index][label_names]
    img_name = df.iloc[index]['filepath']
    img_base_name = os.path.basename(img_name)

    # plot a line graph of the values
    plt.subplot(1, 2, 1)
    sns.lineplot(data=values.T)
    plt.xticks(np.arange(0, len(values), 10), np.arange(0, len(values), 10))
    plt.xlabel('readings')
    plt.ylabel('voltage difference')

    # load and plot the image as a subplot
    img = plt.imread(f'./images/{img_base_name}')
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.axis('off')

    plt.show()

plotVoltageDifferenceLineGraph()