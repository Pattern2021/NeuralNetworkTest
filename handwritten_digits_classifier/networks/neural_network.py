import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from network import Network

class NeuralNetwork:
    """ 
    Base class for network based learning algorithms.

    Parameters:
        Xtr: array_like
            Training data structured as an matrix with N-rows corresponing to datapoint and l-columns representing features.
        Ytr: array_like
            Training data labels. Structured as l-columns with 1 row. Values contained should have class belonging information.
        network_structure: array_like
            Array with information of the structure of the network. len(structure) represents the number of layers, whilst values represents neurons within.
    """
    def __init__(self, Xtr, Ytr, network_structure):
        self.network = Network(network_structure)
        self.w = self.network.w
        self.Xtr, self.Ytr = self.shuffle(Xtr, Ytr)
        self.Xtr = np.append(self.Xtr, np.ones((len(self.Xtr), 1)), axis=1)

    @staticmethod
    def onecolumn(arr):
        return np.append(arr, np.ones((len(arr), 1)), axis=1)

    @staticmethod
    def sigmoid_derivative(x):
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))

    @staticmethod
    @njit
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    @njit
    def hyperbolic_tangent(x):
        return np.tanh(x)

    @staticmethod
    def shuffle(x, y):
        ind = np.arange(len(x))
        np.random.shuffle(ind)
        x = x[ind]
        y = y[0][ind]
        return x, y

def main():
    pass

if __name__ == "__main__":
    main()
