from typing import Literal

from pandas import DataFrame, concat, get_dummies
from sklearn.model_selection import train_test_split


def getEvenDistribution(df: DataFrame, distFeature: Literal['sample', 'shape']) -> DataFrame:
  """
  Get an even distribution of sample types in the dataset.
  
  :param df: The dataset (pandas dataframe).
  :param distFeature: The feature to distribute evenly. It can be 'sample' or 'shape'.
  
  :return: The dataset with an even distribution of sample types.
  """

  # Get the number of samples for each sample type
  sample_counts = df[distFeature].value_counts()

  # Get the maximum number of samples for a sample type
  max_samples = sample_counts.max()

  # Get the sample types with the maximum number of samples
  max_sample_types = sample_counts[sample_counts == max_samples].index

  # Get the sample types with less than the maximum number of samples
  min_sample_types = sample_counts[sample_counts < max_samples].index

  # Get the number of samples to add to each sample type
  samples_to_add = max_samples - sample_counts[min_sample_types]

  # Get the samples to add to each sample type
  samples = []
  for sample_type in min_sample_types:
    samples.append(df[df[distFeature] == sample_type].sample(samples_to_add[sample_type], replace=True))

  # Concatenate the samples to add to the original dataset
  new_df = concat([df] + samples)

  return new_df


def normalise(df: DataFrame) -> DataFrame:
  """
  Normalise the dataset.

  :param df: The dataset (pandas dataframe).
  :return: The normalised dataset.
  """
  # Get the columns where the feature name begins with 'bb_'
  # and 'cc_' but not 'cc_filename' and 'bb_filename'
  columns = df.columns[df.columns.str.startswith('bb_') | df.columns.str.startswith('cc_')]
  columns = columns[~columns.isin(['bb_filename', 'cc_filename'])]

  # Normalise the dataset by dividing each value in the columns by 2e4
  df[columns] = df[columns] / 2e4

  return df


def oneHotEncode(df: DataFrame, feature: str) -> DataFrame:
  """
  One-hot encode a feature in the dataset.

  :param df: The dataset (pandas dataframe).
  :param feature: The feature to one-hot encode.
  :return: The dataset with the one-hot encoded feature.
  """
  feature_column = df[feature]
  # One-hot encode the feature
  df = get_dummies(df, columns=[feature])

  # Add the one-hot encoded feature to the dataset
  df[feature] = feature_column

  return df


def splitDataframeFeature(
  df: DataFrame,
  distFeature: Literal['sample', 'shape'],
  verbose: bool = False
) -> DataFrame:
  """
  Split the dataset into training, testing, and validation datasets.

  :param df: The dataset (pandas dataframe).

  :return: The dataset split into training, testing, and validation datasets.
  """
  # Shuffle the dataset
  df = df.sample(frac=1).reset_index(drop=True)

  df['set'] = 'nan'
  
  # Split the dataframe into features and labels
  X = df.drop(columns=[distFeature])
  y = df[distFeature]

  # Split the dataset into training, testing, and validation datasets
  df_train, df_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
  )
  df_test, df_val, y_test, y_val = train_test_split(
    df_test,
    y_test,
    test_size=0.5,
    random_state=42
  )
  
  # Assign the set labels to the samples
  df.loc[df_train.index, 'set'] = 'train'
  df.loc[df_test.index, 'set'] = 'test'
  df.loc[df_val.index, 'set'] = 'val'

  if verbose:
    print("=============================================")
    print(f"No. samples: {len(df)}")
    print(f"No. training samples: {len(df_train)}\t{len(df_train) / len(df) * 100:.2f}%")
    print(f"No. testing samples: {len(df_test)}\t{len(df_test) / len(df) * 100:.2f}%")
    print(f"No. validation samples: {len(df_val)}\t{len(df_val) / len(df) * 100:.2f}%")
    print("=============================================")

  return df


def splitDataframeMaterial(df: DataFrame) -> tuple[DataFrame, DataFrame]:
  """
  Split the dataset into iron and copper samples.
  
  :param df: The dataset (pandas dataframe).
  :return: A tuple containing the iron and copper samples.
  """
  # Split the dataset into two dataframes, one for iron samples and the other
  # for copper samples. The split is based on the value of the 'sample' column.
  # If the value of the 'sample' column is less than 'H', the sample is iron,
  # otherwise it is copper.
  empty_df = df[df['sample'] == '0']
  iron_df = df[df['sample'] < 'H']
  copper_df = df[df['sample'] >= 'H']
  copper_df = concat([copper_df, empty_df])

  return iron_df, copper_df