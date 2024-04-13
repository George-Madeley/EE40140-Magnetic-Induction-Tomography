import os
import pandas as pd

def cleanUp(removeImages: bool = False, checkMissingImages: bool = False) -> pd.DataFrame:
  """
  The main method that performs the cleanup operations on the dataset.

  :param removeImages: A boolean indicating whether to remove unused images. Default is False.
  :param checkMissingImages: A boolean indicating whether to check for missing images. Default is False.
  :return: The cleaned dataset (pandas dataframe).
  """
  # load in the dataset
  df = getDataframe()

  if removeImages:
    removeUnusedImages(df)

  if checkMissingImages:
    checkForMissingImages(df)

  return df


def getDataframe() -> pd.DataFrame:
  """
  Get the dataset as a pandas dataframe.

  :return: The dataset as a pandas dataframe.
  """

  dfFilepath = os.path.join('data','data_samples.csv')
  if not os.path.exists(dfFilepath):
    samples = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
    df = pd.DataFrame()
    for sample in samples:
      print(f"Reading data_sample-{sample}.csv")
      df = pd.concat([df, pd.read_csv(os.path.join('data', f'data_sample-{sample}.csv'))])

    print(f"Combined dataframe has {len(df)} rows")
    # Save the combined dataframe to a new csv file
    df.to_csv(dfFilepath, index=False)

  else:
    # Read the combined dataframe from the csv file. The
    # file does not have an index column therefore set
    # index_col=False
    df = pd.read_csv(dfFilepath, index_col=False)
    
  df['sample'] = df['sample'].astype(str)
  df['bb_filename'] = df['bb_filename'].astype(str)
  df['cc_filename'] = df['cc_filename'].astype(str)

  return df


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

def deleteImages():
  """
  Delete all the images in the processed directory that are not in the 480x640
  directory.
  """

  # Get the list of images in the 480x640 directory
  image_names = os.listdir(os.path.join('images', 'processed', '480x640'))

  factors = [2, 4, 5, 8, 10, 16, 20, 32]

  # Delete all the images in the processed directory that are not in the 480x640 directory
  for factor in factors:
    fileDir = os.path.join('images', 'processed', f'{480//factor}x{640//factor}')
    processed_images = os.listdir(fileDir)
    for image_name in processed_images:
      if image_name not in image_names:
        print(f"Deleting {image_name} from {480//factor}x{640//factor}")
        os.remove(os.path.join(fileDir, image_name))