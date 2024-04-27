from .IModel import IModel

from .classification import DecisionTree
from .classification import KNearestNeighbors
from .classification import NearestCentroid
from .classification import NeuralNetwork as ClassificationNeuralNetwork
from .classification import RandomForest
from .classification import StochasticGradientDescent
from .classification import SupportVectorMachine



from .generation import GenerativeAdversarialNetwork as GAN
from .generation import VariationalAutoencoder as VAE
from .generation import NeuralNetwork as GenerationNeuralNetwork
from .generation import UNetwork as UNet
