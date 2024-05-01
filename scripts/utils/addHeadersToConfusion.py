import pandas as pd
import os

def addHeadersToConfusion():
  directory = os.path.join('results', 'ConfusionMatrics')
  for filename in os.listdir(directory):
    df = pd.read_csv(os.path.join(directory, filename))

    if 'iron' in filename:
      headers = ['N', 'S', 'T', 'X', '_']
    elif 'copper' in filename:
      headers = ['0', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

    # Change the headers
    df.columns = headers

    # Add a new column to the start of the dataframe
    df.insert(0, 'labels', headers)

    # Save the new dataframe
    df.to_csv(os.path.join(directory, filename), index=False)

if __name__ == '__main__':
  addHeadersToConfusion()