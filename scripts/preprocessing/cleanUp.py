import os
from typing import Literal

from sklearn.model_selection import train_test_split
import pandas as pd

class cleanUp:
  """
  A class that provides methods for cleaning up a dataset and handling images.
  """

  @staticmethod
  def main(removeImages: bool = False, checkMissingImages: bool = False) -> pd.DataFrame:
    """
    The main method that performs the cleanup operations on the dataset.

    :param removeImages: A boolean indicating whether to remove unused images. Default is False.
    :param checkMissingImages: A boolean indicating whether to check for missing images. Default is False.
    :return: The cleaned dataset (pandas dataframe).
    """
    # load in the dataset
    df = cleanUp.getDataframe()

    if removeImages:
      cleanUp.removeUnusedImages(df)

    if checkMissingImages:
      cleanUp.checkForMissingImages(df)

    return df

  @staticmethod
  def getDataframe(df_name: str = None) -> pd.DataFrame:
    """
    Get the dataset as a pandas dataframe.

    :param df_name: The name of the dataset file. Default is None.
    :return: The dataset as a pandas dataframe.
    """
    if df_name is not None and os.path.exists(df_name):
      df = pd.read_csv(df_name, index_col=False)
      df['sample'] = df['sample'].astype(str)
      return df

    if not os.path.exists('./data/data_samples.csv'):
      samples = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
      dataframePrefix = './data/data_sample-'
      dataframeSuffix = '.csv'
      df = pd.DataFrame()
      for sample in samples:
        print(f"Reading {dataframePrefix + sample + dataframeSuffix}")
        df = pd.concat([df, pd.read_csv(dataframePrefix + sample + dataframeSuffix)])

      print(f"Combined dataframe has {len(df)} rows")
      # Save the combined dataframe to a new csv file
      df.to_csv('./data/data_samples.csv', index=False)

    else:
      # Read the combined dataframe from the csv file. The
      # file does not have an index column therefore set
      # index_col=False
      df = pd.read_csv('./data/data_samples.csv', index_col=False)

      # All the values in the sample column are of type string.
      # Convert them to type string.
      df['sample'] = df['sample'].astype(str)

    return df

  @staticmethod
  def removeUnusedImages(df: pd.DataFrame) -> None:
    """
    Remove all the images that are not used by the dataset

    :param df: The dataset (pandas dataframe).
    """
    # Set the path to the images
    path = os.path.join(os.getcwd(), 'images', 'original')

    # Create a new directory called 'used' to store the unused images
    used_path = os.path.join(path, 'used')
    os.makedirs(used_path, exist_ok=True)

    # Get all the unique image names in the column 'bb_filename'
    bb_image_names = df['bb_filename'].unique()

    # Get all the unique image names in the column 'cc_filename'
    cc_image_names = df['cc_filename'].unique()

    # Combine the two lists of image names
    image_names = list(bb_image_names) + list(cc_image_names)

    # Move all the images in image_names list to the 'used' directory
    for image_name in image_names:
      if os.path.exists(os.path.join(path, image_name)):
        print(f"Moving {image_name} to 'used' directory")
        os.rename(os.path.join(path, image_name), os.path.join(used_path, image_name))
      else:
        print(f"\033[91mERROR: Image {image_name} does not exist in 'original' directory\033[0m")

    # Delete each image in the 'original' directory that begins with 'snapshot_'
    count = 0
    for image_name in os.listdir(path):
      if image_name.startswith('snapshot_'):
        count += 1
        print(f"Deleting {image_name}")
        os.remove(os.path.join(path, image_name))
    print(f"Deleted {count} unused images")

    # Move all the images in the 'used' directory back to the 'original' directory
    for image_name in os.listdir(used_path):
      print(f"Moving {image_name} back to 'original' directory")
      os.rename(os.path.join(used_path, image_name), os.path.join(path, image_name))

    # Remove the 'used' directory
    os.rmdir(used_path)

  @staticmethod
  def checkForMissingImages(df: pd.DataFrame) -> None:
    """
    Check for missing images in the dataset.

    :param df: The dataset (pandas dataframe).
    """
    # Get all the unique image names in the column 'bb_filename'
    bb_image_names = df['bb_filename'].unique()

    # Get all the unique image names in the column 'cc_filename'
    cc_image_names = df['cc_filename'].unique()

    # Combine the two lists of image names
    image_names = list(bb_image_names) + list(cc_image_names)

    # Set the path to the images
    path = os.path.join(os.getcwd(), 'images', 'original')

    # Check if any image in the image_names list does not exist in the 'original' directory
    missing_images = []
    for image_name in image_names:
      if not os.path.exists(os.path.join(path, image_name)):
        missing_images.append(image_name)

    if len(missing_images) > 0:
      print(f"\033[91mERROR: The following images are missing from the 'original' directory: \033[0m")
      for image_name in missing_images:
        print(image_name)

  @staticmethod
  def splitDataframeMaterial(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    copper_df = pd.concat([copper_df, empty_df])

    return iron_df, copper_df

  @staticmethod
  def getEvenDistribution(df: pd.DataFrame, distFeature: Literal['sample', 'shape']) -> pd.DataFrame:
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
    new_df = pd.concat([df] + samples)

    return new_df

  @staticmethod
  def splitDataframe(
    df: pd.DataFrame,
    distFeature: Literal['sample', 'shape'],
    verbose: bool = False
  ) -> pd.DataFrame:
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