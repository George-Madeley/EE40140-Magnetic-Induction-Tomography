import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plotVoltageDifferenceLineGraph():
    # load data from csv with headers
    df = pd.read_csv('./input_data.csv')

    # get the names of the sensor columns
    label_names = df.filter(regex='sensor_\d+').columns
    values = df.iloc[-1][label_names]

    # plot a line graph of the values
    sns.lineplot(data=values.T)
    plt.xticks(np.arange(0, len(values), 10), np.arange(0, len(values), 10))
    plt.xlabel('readings')
    plt.ylabel('voltage difference')
    plt.show()

plotVoltageDifferenceLineGraph()