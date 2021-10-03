import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from neural_network import *

class Data_preprocesser(NeuralNetwork):
    """
    Preprocesses data and formats to correct layout, ready to feed into children of neural network base class. Also inherits from Neural network base clas.

    Parameters:
        *Xtr: array_like
            Training data from unique classes.
        shuffle: boolean, optional
            Shuffling data using random order.
    """
    def __init__(self, *Xtr, shuffle=True):
        self.Xtr = np.asarray(Xtr)
        pass
        


def main():
    a1 = np.random.uniform(0, 1, size=(10, 2))
    a2 = np.random.uniform(1, 2, size=(10, 2))
    Data_preprocesser(a1, a2)

if __name__ == "__main__":
    main()
