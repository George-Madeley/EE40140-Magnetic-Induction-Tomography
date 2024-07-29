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

  num_frames = args.get('--numFrames', 1)
  frequency = args.get('--frequency', 20000)
  gain = args.get('--gain', 7)

  print(f'MIT Operating at {frequency}Hz and gain {gain} with {num_frames} frames.')
  print("Reading background signal...")
  bb_data = readSensor(
    frequency=frequency,
    gain=gain,
    num_frames=num_frames
  )
  print("Reading coil signal...")

  fig, axs = plt.subplots(1, 2)
  ax1, ax2 = axs

  def animate(i):
    '''
    Animation function.
    '''
    cc_data = readSensor(
      frequency=frequency,
      gain=gain,
      num_frames=num_frames
    )

    # concatenate the data
    data = np.concatenate((cc_data, bb_data)).astype(np.float32)

    # convert the data to a tensor
    data = torch.tensor(data, dtype=torch.float32).to(model.device)

    # Add a batch dimension
    data = data.unsqueeze(0)

    # generate the image
    generated_image = model.predict(data)
    generated_image = generated_image.squeeze(0).squeeze(0).detach().cpu().numpy()
    height, width = generated_image.shape
    diff = (width - height) // 2
    # remove horizontal padding
    generated_image = generated_image[:, diff:-diff]

    # plot cc_data and bb_data on the first axis then the generated image on the
    # second axis
    ax1.clear()
    ax1.plot(bb_data, label='bb_data', color='darkblue', alpha=0.3)
    ax1.plot(cc_data, label='cc_data', color='darkblue')
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

  args_types = {
    '--generator': str,
    '--labelName': str,
    '--noise': bool,
    '--downScaleFactor': int,
    '--batchSize': int,
    '--learningRate': float,
    '--maxEpoch': int,
    '--perPixelLoss': bool,
    '--oneHotEncode': bool,
    '--toSave': bool,
    '--numFrames': int,
    '--frequency': int,
    '--gain': int,
  }

  arg_longs = {
    '-g': '--generator',
    '-ln': '--labelName',
    '-n': '--noise',
    '-dsf': '--downScaleFactor',
    '-bs': '--batchSize',
    '-lr': '--learningRate',
    '-me': '--maxEpoch',
    '-ppl': '--perPixelLoss',
    '-ohe': '--oneHotEncode',
    '-ts': '--toSave',
    '-nf': '--numFrames',
    '-f': '--frequency',
  }

  argv = sys.argv[1:]
  if len(argv) == 0:
    return {}
  
  # if the first element is '-m', remove it
  if argv[0] == '-m':
    argv = argv[1:]
  
  if len(argv) % 2 != 0:
    raise ValueError("Invalid number of arguments")
  
  # get every every even index and every odd index
  args = {argv[i]: argv[i + 1] for i in range(0, len(argv), 2)}

  # check that all keys start with '--' or '-'. To do this, copy the keys to a
  # list and iterate through the list. If the key does not start with '--' or '-'
  # raise a ValueError.
  keys = list(args.keys())
  for key in keys:
    if not key.startswith('--') and not key.startswith('-'):
      raise ValueError(f'Invalid argument key: {key}')
    
    if key not in args_types and key not in arg_longs:
      raise ValueError(f'Invalid argument key: {key}')
    
    if key.startswith('-') and not key.startswith('--'):
      args[arg_longs[key]] = args.pop(key)

  # check that all values are of the correct type
  for key, value in args.items():
    if key not in args_types:
      continue
    
    try:
      args[key] = args_types[key](value)
    except ValueError as e:
      raise ValueError(f"Invalid value for {key}: {value}")

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