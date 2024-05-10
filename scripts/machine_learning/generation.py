import os
from typing import List, Literal


from models.generation import *

import pandas as pd
import time

def runModels():
  """
  Run the generation models
  """

  material = 'unknown'
  labelName = 'sample'
  batchSize = 45

  if material == 'iron' and labelName == 'shape':
    pass
  elif material == 'copper' and labelName == 'sample':
    pass
  elif material == 'unknown' and labelName == 'sample':
    pass
  else:
    raise ValueError(f"Material {material} and label {labelName} not correct combination")

  if material == 'iron':
    fixedIndices = [1342, 1234, 1239, 944, 609, 417, 1235, 120]
  elif material == 'copper':
    fixedIndices = [1342, 3457, 1238, 610, 123, 1236, 948, 419]
  else:
    print(f"Material {material} not found")
    fixedIndices = [1 for _ in range(8)]

  # df_train, df_test, df_val = getData(material)

  # commonFactors = getCommonFactors(df_train.shape[0], df_test.shape[0])

  # if batchSize not in commonFactors:
  #   raise ValueError(f"Common factor {batchSize} not found in {commonFactors}")


  NNs = {
    'NN1': [480],
    'NN2': [960],
    'NN3': [1200],
    'NN4': [2400],
    'NN5': [480, 960],
    'NN6': [480, 2400],
    'NN7': [1200,2400],
  }

  CNNs = {
    'CNN1': [2],
    'CNN2': [4],
    'CNN3': [5],
    'CNN4': [10],
    'CNN5': [2, 4],
    'CNN6': [2, 10],
    'CNN7': [5, 10],
  }

  GANs = {
    'GAN1': {
      'discriminator': [64],
      'generator': [480]
    },
    'GAN2': {
      'discriminator': [64],
      'generator': [960]
    },
    'GAN3': {
      'discriminator': [15, 240],
      'generator': [1200]
    },
    'GAN4': {
      'discriminator': [15, 240],
      'generator': [2400]
    },
    'GAN5': {
      'discriminator': [20, 320],
      'generator': [480, 960]
    },
    'GAN6': {
      'discriminator': [20, 320],
      'generator': [480, 2400]
    },
    'GAN7': {
      'discriminator': [20, 400],
      'generator': [1200, 2400]
    },
  }

  DCGANs = {
    'DCGAN1': {
      'discriminator': [2],
      'generator': [2]
    },
    'DCGAN2': {
      'discriminator': [4],
      'generator': [4]
    },
    'DCGAN3': {
      'discriminator': [2, 4, 2],
      'generator': [2, 4, 2]
    },
    'DCGAN4': {
      'discriminator': [2, 6, 2],
      'generator': [2, 6, 2]
    },
    'DCGAN5': {
      'discriminator': [16, 32, 16],
      'generator': [16, 32, 16]
    },
  }

  VAEs = {
    'VAE1': {
      'encoder': [48, 12, 4],
      'decoder': [10, 60, 480]
    },
    'VAE2': {
      'encoder': [48, 12, 4],
      'decoder': [20, 160, 960]
    },
    'VAE3': {
      'encoder': [120, 40, 10],
      'decoder': [10, 60, 480]
    },
    'VAE4': {
      'encoder': [120, 40, 10],
      'decoder': [20, 160, 960]
    },
    'VAE5': {
      'encoder': [40, 8, 4],
      'decoder': [8, 48, 480]
    },
    'VAE6': {
      'encoder': [40, 8, 4],
      'decoder': [12, 60, 480]
    },
  }

  UNNs = {
    'UNN1': [5],
    'UNN2': [5, 2],
    'UNN3': [5, 2, 2],
    'UNN4': [5, 4],
    'UNN5': [2],
    'UNN6': [2, 2],
    'UNN7': [2, 5],
    'UNN8': [2, 2, 5],
    'UNN9': [4, 5],
    'UNN10': [10],
    'UNN11': [10, 2],
  }

  ResNets = {
    'ResNet': []
  }

  defaultArgs = {
    'labelName': labelName,
    'noise': True,
    # 'downScaleFactor': 8,
    'batchSize': batchSize,
    'learningRate': 0.0001,
    'maxEpoch': 1,
    'perPixelLoss': True,
    'oneHotEncode': True,
    'toSave': True,
  }

  best_UNN = list(UNNs.items())[0]

  models = [
    UNN(
      name=best_UNN[0],
      structure=best_UNN[1],
      **defaultArgs
    )
  ]

  # predictUnknows(material, models, batchSize)

  down_scale_factors = [32, 16, 8, 4]
  files = [
    'UNN1 - RUhMoXnXrY',
    'UNN1 - aQzGkfHPos',
    'UNN1 - ltxbOJIHIV',
    'UNN1 - eJgQvcwlUb',
  ]

  for down_scale_factor, file in zip(down_scale_factors, files):
    model = UNN(
      name=best_UNN[0],
      structure=best_UNN[1],
      downScaleFactor=down_scale_factor,
      loadFile=file,
      **defaultArgs
    )
    predictUnknows(material, [model], batchSize)


def predictUnknows(material, models, batchSize=45):
    df_unknown_path = os.path.join('data', f'data_samples_{material}.csv')
    df_unknown = pd.read_csv(df_unknown_path)

    num_to_keep = (df_unknown.shape[0] // batchSize) * batchSize
    df_unknown = df_unknown.head(num_to_keep)

    for model in models:
      print(f"Running {model.__class__.__name__}")
      model.loadModel()

      unknownLoader = model.getLoader(df_unknown)

      imgDir = os.path.join('images', 'generated', 'unknown')
      os.makedirs(imgDir, exist_ok=True)

      # start perf timer here
      start_time = time.perf_counter_ns()

      for batch_idx, batch in enumerate(unknownLoader):
        realImageSamples, signalSamples, signalLabels, _ = batch

        realImageSamples = realImageSamples.to(device=model.device)
        signalSamples = signalSamples.to(device=model.device)

        generatedImageSamples = model.predict(signalSamples)
        errorImages = model.calculatePerPixelLoss(realImageSamples, generatedImageSamples)

      # end perf timer here
      end_time = time.perf_counter_ns()

      print(f"Downscale factor: {model.downScaleFactor}")
      print(f"Time taken: {(end_time - start_time)} ns")
      print(f"Avg time per sample: {(end_time - start_time) / num_to_keep} ns\n\n")

        # for i in range(realImageSamples.shape[0]):
        #   realImage = realImageSamples[i].cpu().detach().numpy()
        #   generatedImage = generatedImageSamples[i].cpu().detach().numpy()
        #   errorImage = errorImages[i].cpu().detach().numpy()

        #   save_id = i + batch_idx * model.batchSize

        #   model.plotImages(
        #   imgDir,
        #   realImage,
        #   f'unknown_original - {str(save_id).zfill(2)}.png',
        #   subtitle='Original',
        #   numCols=1
        # )
        #   model.plotImages(
        #   imgDir,
        #   generatedImage,
        #   f'unknown_generated - {str(save_id).zfill(2)}.png',
        #   subtitle='Generated',
        #   numCols=1
        # )
        #   model.plotImages(
        #   imgDir,
        #   errorImage,
        #   f'unknown_error - {str(save_id).zfill(2)}.png',
        #   subtitle='Error',
        #   palette='viridis',
        #   colorBar=True,
        #   colorRange=(-1, 1),
        #   numCols=1
        # )

  

def getData(material: Literal['iron', 'copper'] = 'iron'):
  """
  Get the data for the specified material

  :param material: the material to get the data for

  :return: the data
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
  a (int): The first number.
  b (int): The second number.

  Returns:
  list: A list of common factors between a and b.

  """
  factors: List[int] = []
  for i in range(1, min(a, b) + 1):
    if a % i == 0 and b % i == 0:
      factors.append(i)
  return factors

if __name__ == "__main__":
  runModels()