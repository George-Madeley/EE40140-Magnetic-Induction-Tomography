import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import torch

from machine_learning.models.generation import *

import os
import sys
import json
import csv

def MIT():
  '''
  Main function of the script. It is responsible for running the script.
  '''

  args = getArgs()
  model = getGenerator(**args)

  # data = readSensor()

  fig = plt.figure()
  ax1 = fig.add_subplot(1, 2, 1)
  ax2 = fig.add_subplot(1, 2, 2)

  def animate(i):
    '''
    Animation function.
    '''
    cc_data, bb_data = getDataFromFile(i)
    # concatenate the data
    data = np.concatenate((cc_data, bb_data)).astype(np.float32)

    # convert the data to a tensor
    data = torch.tensor(data, dtype=torch.float32).to(model.device)

    # Add a batch dimension
    data = data.unsqueeze(0)

    # generate the image
    generated_image = model.predict(data)
    generated_image = generated_image.squeeze(0).squeeze(0).detach().cpu().numpy()


    # plot cc_data and bb_data on the first axis then the generated image on the
    # second axis
    ax1.clear()
    ax1.plot(cc_data, label='cc_data')
    ax1.plot(bb_data, label='bb_data')
    ax1.legend()
    ax1.set_title('Signal')
    ax1.set_xlabel('Coil')
    ax1.set_ylabel('Voltage Differences')

    ax2.clear()
    ax2.imshow(generated_image, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Generated Image')


  ani = animation.FuncAnimation(fig, animate)
  plt.show()


def getArgs():
  '''
  Arguments of the script a retrieved here and returned as a dictionary.
  Arguments come in the form of --argName argValue or -argName argValue.

  Returns:
    dict: Dictionary containing the arguments of the script.
  '''

  argv = sys.argv[1:]
  if len(argv) == 0:
    return {}
  
  if len(argv) % 2 != 0:
    raise ValueError("Invalid number of arguments")
  
  # get every every even index and every odd index
  args = {argv[i]: argv[i + 1] for i in range(0, len(argv), 2)}

  # check that all keys start with '--' or '-'
  for key in args.keys():
    if not key.startswith('--') and not key.startswith('-'):
      raise ValueError(f'Invalid argument key: {key}')

  return args

def getGenerator(**kwargs):
  '''
  Gets the image generator.
  
  Returns:
    generator: Generator that generates images.
  '''

  if '--generator' not in kwargs and '-g' not in kwargs:
    raise ValueError("No generator specified")
  
  # get the generator filepath
  file_path = kwargs.get('--generator', kwargs.get('-g'))

  # check if the file exists
  if not os.path.exists(file_path):
    raise ValueError(f"File {file_path} does not exist")
  
  # split the file path and the extension
  file_path, ext = os.path.splitext(file_path)

  # check if the extension is valid
  if ext != '.pt':
    raise ValueError(f"Invalid extension: {ext}")

  # get the directory and file name
  directory, file_name = os.path.split(file_path)

  # split the file name ' - '
  file_name = file_name.split(' - ')

  if len(file_name) != 2:
    raise ValueError(f"Invalid file name: {file_name}")
  
  # get the model anme and the model version
  model_version, model_id = file_name

  # get the model structures
  with open(os.path.join('scripts', 'machine_learning', 'model_structures.json'), 'r') as f:
    model_structures = json.load(f)

  # check if the model name is in the model structures
  if model_version not in model_structures.keys():
    raise ValueError(f"Model {model_version} not in model structures {model_structures.keys()}")
  
  model_structure = model_structures[model_version]

  model_classes = {
    'CNN': NN,
    'DCGAN': GAN,
    'GAN': GAN,
    'NN': NN,
    'ResNet': ResNet,
    'UNN': UNN,
    'VAE': VAE,
  }

  # get a list of all the models that start with the model name
  model_initials = [model for model in model_classes.keys() if model_version.startswith(model)]
  if len(model_initials) != 1:
    raise ValueError(f"Invalid model initials: {model_initials}")
  model_class = model_classes[model_initials[0]]

  default_model_args = {
    'labelName': kwargs.get('--labelName', kwargs.get('-ln', 'material')),
    'noise': kwargs.get('--noise', kwargs.get('-n', True)),
    'downScaleFactor': kwargs.get('--downScaleFactor', kwargs.get('-dsf', 8)),
    'batchSize': kwargs.get('--batchSize', kwargs.get('-bs', 45)),
    'learningRate': kwargs.get('--learningRate', kwargs.get('-lr', 0.0001)),
    'maxEpoch': kwargs.get('--maxEpoch', kwargs.get('-me', 1000)),
    'perPixelLoss': kwargs.get('--perPixelLoss', kwargs.get('-ppl', True)),
    'oneHotEncode': kwargs.get('--oneHotEncode', kwargs.get('-ohe', True)),
    'toSave': kwargs.get('--toSave', kwargs.get('-ts', False)),
  }

  model = model_class(
    name=model_version,
    structure=model_structure,
    **default_model_args,
    loadFile=file_path + ext,
  )
  model.loadModel()

  return model

def readSensor(
    frequency: int = 20000,
    gain: int = 7,
    num_frames: int = 1,
):
  '''
  Reads the sensor data from the sensor.
  '''

  # Run the sensor
  sensor_dir = os.path.join('scripts', 'data_collections')
  os.chdir(sensor_dir)
  os.system(f'MIT_Multi_Frame.exe {frequency} {gain} {num_frames}')
  os.chdir(os.path.join('..', '..'))

  # Read the data
  data_path = os.path.join('scripts', 'data_collections', 'data.csv')
  with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    data = list(reader)

  data = data[0][:-1]
  
  # convert the data to a numpy array
  data = np.array(data, dtype=np.float32)

  # reshape the data
  data = data.reshape(num_frames, 120).T

  # average the data on the second axis
  data = np.mean(data, axis=1)

  # set any value bigger than 2e4 to 0
  data[data > 2e4] = 0

  # normalise the data
  data = data / 2e4

  return data
  
def getDataFromFile(index: int = 0):
  df_path = os.path.join('data', 'data_samples_iron.csv')
  df = pd.read_csv(df_path)

  # get columns that begin with 'cc_' but not 'cc_filename'
  columns = [column for column in df.columns if column.startswith('cc_') and column != 'cc_filename']
  cc_data = df.iloc[index][columns].values

  
  columns = [column for column in df.columns if column.startswith('bb_') and column != 'bb_filename']
  bb_data = df.iloc[index][columns].values

  return cc_data, bb_data


if __name__ == '__main__':
  MIT()