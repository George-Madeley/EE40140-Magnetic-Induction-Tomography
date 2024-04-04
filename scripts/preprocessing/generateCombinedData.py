import pandas as pd

def main():
  filePrefix = 'data_sample-'
  fileSuffix = '.csv'

  # read data from sample 0
  df = pd.read_csv(filePrefix + '0' + fileSuffix)

  samples = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
  for sample in samples:
    print('Processing sample ' + sample)
    df = pd.concat([df, pd.read_csv(filePrefix + sample + fileSuffix)])

  df.to_csv('data_samples.csv', index=False)

if __name__ == '__main__':
  main()
