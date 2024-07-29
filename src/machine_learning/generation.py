import os
import json
import sys
from typing import Dict, List, Literal, Tuple, Union
import pandas as pd
import time
from models.generation import NN, GAN, ResNet, UNN, VAE


def trainModels(
  material: str,
  labelName: str,
  batchSize: int,
  fixedIndices: List[int],
  splitData: bool = True
) -> None:
  """
  Run the generation models.

  Parameters:
  - material (str): The material to run the models on.
  - labelName (str): The label name for the models.
  - batchSize (int): The batch size for processing the data.
  - fixedIndices (List[int]): List of fixed indices.

  Returns:
  None
  """

  if splitData:
    df_train, df_test, df_val = getData(material)

    commonFactors = getCommonFactors(df_train.shape[0], df_test.shape[0])

    if batchSize not in commonFactors:
      raise ValueError(
        f"Common factor {batchSize} not found in {commonFactors}")
  else:
    df = getData(material)

  # get the model_structures.json file
  models: Dict[str, Dict[str, Union[NN, GAN, ResNet, UNN, VAE]]] = {
    'CNN': {},
    'DCGAN': {},
    'GAN': {},
    'NN': {},
    'ResNet': {},
    'UNN': {},
    'VAE': {},
  }
  filepath = os.path.join(
      'scripts',
      'machine_learning',
      'model_structures.json')
  with open(filepath, 'r') as f:
    model_structures = json.load(f)

    for model_initial in models.keys():
      # Get a dictionary of all the models in model_structures that start with
      # the model_initial
      model_names = [model for model in model_structures.keys()
                     if model.startswith(model_initial)]

      # add each model to the models dictionary
      for model_name in model_names:
        models[model_initial][model_name] = model_structures[model_name]

  defaultArgs = {
    'labelName': labelName,
    'noise': True,
    'downScaleFactor': 8,
    'batchSize': batchSize,
    'learningRate': 0.0001,
    'maxEpoch': 1000,
    'perPixelLoss': True,
    'oneHotEncode': True,
    'toSave': True,
  }

  for model_name, model_versions in models.items():
    for version_name, version_structure in model_versions.items():
      if model_name == 'CNN':
        model = NN(
          name=version_name,
          structure=version_structure,
          conv=True,
          **defaultArgs
        )
      elif model_name == 'DCGAN':
        model = GAN(
          name=version_name,
          structure=version_structure,
          convD=True,
          convG=True,
          **defaultArgs
        )
      elif model_name == 'GAN':
        model = GAN(
          name=version_name,
          structure=version_structure,
          **defaultArgs
        )
      elif model_name == 'NN':
        model = NN(
          name=version_name,
          structure=version_structure,
          **defaultArgs
        )
      elif model_name == 'ResNet':
        model = ResNet(
          name=version_name,
          structure=version_structure,
          **defaultArgs
        )
      elif model_name == 'UNN':
        model = UNN(
          name=version_name,
          structure=version_structure,
          **defaultArgs
        )
      elif model_name == 'VAE':
        model = VAE(
          name=version_name,
          structure=version_structure,
          **defaultArgs
        )
      model.run(
        df_train,
        df_test,
        df_val,
        fixedIndices
      )


def runModel(
  filePath: str,
  model_name: str,
  model_version: str,
  model_file: str,
  batchSize: int = 45
) -> None:
  """
  Run the specified models on the given material data.

  Parameters:
  - filePath (str): The file path of the material data.
  - model_name (str): The name of the model.
  - model_version (str): The version of the model.
  - model_file (str): The file name of the model.
  - batchSize (int): The batch size for processing the data. Default is 45.

  Returns:
  None
  """
  df_unknown = pd.read_csv(filePath)

  num_to_keep = (df_unknown.shape[0] // batchSize) * batchSize
  df_unknown = df_unknown.head(num_to_keep)

  filepath = os.path.join(
      'scripts',
      'machine_learning',
      'model_structures.json')
  with open(filepath, 'r') as f:
    model_structures = json.load(f)

    version_structure = model_structures.get(
      model_file, {}).get(model_version, None)

    if version_structure is None:
      raise ValueError(
        f"Model {model_file} version {model_version} not found in {filepath}")

  defaultArgs = {
    'labelName': labelName,
    'noise': True,
    'downScaleFactor': 8,
    'batchSize': batchSize,
    'learningRate': 0.0001,
    'maxEpoch': 1000,
    'perPixelLoss': True,
    'oneHotEncode': True,
    'toSave': True,
  }

  if model_name == 'CNN':
    model = NN(
      name=model_version,
      structure=version_structure,
      loadFile=model_file,
      conv=True,
      **defaultArgs
    )
  elif model_name == 'DCGAN':
    model = GAN(
      name=model_version,
      structure=version_structure,
      loadFile=model_file,
      convD=True,
      convG=True,
      **defaultArgs
    )
  elif model_name == 'GAN':
    model = GAN(
      name=model_version,
      structure=version_structure,
      loadFile=model_file,
      **defaultArgs
    )
  elif model_name == 'NN':
    model = NN(
      name=model_version,
      structure=version_structure,
      loadFile=model_file,
      **defaultArgs
    )
  elif model_name == 'ResNet':
    model = ResNet(
      name=model_version,
      structure=version_structure,
      loadFile=model_file,
      **defaultArgs
    )
  elif model_name == 'UNN':
    model = UNN(
      name=model_version,
      structure=version_structure,
      loadFile=model_file,
      **defaultArgs
    )
  elif model_name == 'VAE':
    model = VAE(
      name=model_version,
      structure=version_structure,
      loadFile=model_file,
      **defaultArgs
    )

  print(f"Running {model.__class__.__name__}")
  model.loadModel()

  unknownLoader = model.getLoader(df_unknown)

  imgDir = os.path.join('images', 'generated', 'unknown')
  os.makedirs(imgDir, exist_ok=True)

  for batch_idx, batch in enumerate(unknownLoader):
    realImageSamples, signalSamples, signalLabels, _ = batch

    realImageSamples = realImageSamples.to(device=model.device)
    signalSamples = signalSamples.to(device=model.device)

    generatedImageSamples = model.predict(signalSamples)
    errorImages = model.calculatePerPixelLoss(
      realImageSamples, generatedImageSamples)

    for i in range(realImageSamples.shape[0]):
      realImage = realImageSamples[i].cpu().detach().numpy()
      generatedImage = generatedImageSamples[i].cpu().detach().numpy()
      errorImage = errorImages[i].cpu().detach().numpy()

      save_id = i + batch_idx * model.batchSize

      model.plotImages(
        imgDir,
        realImage,
        f'unknown_original - {str(save_id).zfill(2)}.png',
        subtitle='Original',
        numCols=1
      )
      model.plotImages(
        imgDir,
        generatedImage,
        f'unknown_generated - {str(save_id).zfill(2)}.png',
        subtitle='Generated',
        numCols=1
      )
      model.plotImages(
        imgDir,
        errorImage,
        f'unknown_error - {str(save_id).zfill(2)}.png',
        subtitle='Error',
        palette='viridis',
        colorBar=True,
        colorRange=(-1, 1),
        numCols=1
      )


def getData(material: Literal['iron',
                              'copper'] = 'iron') -> Tuple[pd.DataFrame,
                                                           pd.DataFrame,
                                                           pd.DataFrame]:
  """
  Get the data for the specified material.

  Parameters:
  - material (Literal['iron', 'copper']): The material to get the data for.

  Returns:
  Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The data as a tuple of dataframes.
  """
  dataFilePath = os.path.join('data', f'data_samples_{material}.csv')
  df = pd.read_csv(dataFilePath)
  df_train = df[df['set'] == 'train']
  df_test = df[df['set'] == 'test']
  df_val = df[df['set'] == 'val']

  return df_train, df_test, df_val


def getCommonFactors(a: int, b: int) -> List[int]:
  """
  Returns a list of common factors between two numbers within a specified range.

  Parameters:
  - a (int): The first number.
  - b (int): The second number.

  Returns:
  List[int]: A list of common factors between a and b.
  """
  factors: List[int] = []
  for i in range(1, min(a, b) + 1):
    if a % i == 0 and b % i == 0:
      factors.append(i)
  return factors


if __name__ == "__main__":
  argv = sys.argv[2:]
  material = argv[0]
  labelName = argv[1]
  batchSize = int(argv[2])
  model_name = argv[3]
  model_file = argv[4]

  if material == 'iron':
    fixedIndices = [7, 22, 35, 901, 219, 220, 11, 12]
  elif material == 'copper':
    fixedIndices = [419, 123, 1238, 700, 610, 948, 949, 1236]
  else:
    print(f"Material {material} not found")
    fixedIndices = [1 for _ in range(8)]

  trainModels(
    material=material,
    labelName=labelName,
    batchSize=batchSize,
    fixedIndices=fixedIndices,
    splitData=True
  )
