import numpy as np
import matplotlib.pyplot as plt

class Node:
    """ 
    Structure class of Neural Network.

    Parameters:
        neurons: array_like
            Number of neurons per layer. First element is input layer, last is output layer, rest is hidden layers. Array must have shape (n,) and consist of integers
    """
    def __init__(self, neurons=np.array([3, 7, 5, 2])):
        try:
            if isinstance(neurons, (np.ndarray, list)):

                # Separating and setting variables
                self.neurons = neurons
                self.layers = len(neurons)
                self.total_neurons = np.sum(neurons)
                self.input_neurons = neurons[0]
                self.output_neurons = neurons[-1]
                self.hidden_layers = neurons[1:-1]
                self.hidden_neurons = np.sum(self.hidden_layers)
                self.max_neurons_in_layer = neurons[np.argmax(neurons)]

                # Setting colormap could also use prism
                cmap = plt.cm.gist_rainbow
                self.colors = cmap(np.abs(np.linspace(-1, 1, self.total_neurons)))
            else:
                raise TypeError("Parameter must be an numpy.ndarray, not {}.".format(type(neurons)))
        except TypeError as e:
            print(e)
        
    def plot_structure(self):
        """ 
        Method which plots the structure of given network.
        """

        self.fig, self.axs = plt.subplots(1, 1, figsize=(10, 7))

        coordinates = []

        # Loop through every layer.
        for index, layer in enumerate(self.neurons):

            # Setting an y position for each neuron based on number of neurons in each layer
            shape = np.arange(0, layer)

            # Normalising position neurons in each layer
            structure = shape - np.mean(shape)

            # Casting dimensions from (n,) -> (n, 1)
            structure = np.atleast_2d(structure).T

            # Setting x position for neurons of each layer as the iterative index + 1
            xarr = np.ones_like(structure) * index + 1

            # Merging arrays in such a way that form is [xpos, ypos] x n
            coordinate = np.append(xarr, structure, axis=1)

            for coord in coordinate:
                coordinates.append(coord)

        coordinates = np.asarray(coordinates)


        for i, coords in enumerate(coordinates):

            # Forces index only to affect for i less than the amount of layers - 1
            if i < len(coordinates) - self.output_neurons:

                # Finds all coordinates of only next layer
                draw_to = coordinates[np.where((coords[0] < coordinates[:, 0]) & (coordinates[:, 0] <= coords[0] + 1))]
            
            for vec in draw_to:

                # Draws lines from each neuron to the next neuron if the coordinates passed the previous condition
                xarr = np.array([coords[0], vec[0]])
                yarr = np.array([coords[1], vec[1]])

                self.axs.plot(xarr, yarr, color=self.colors[i], zorder=1)

        # Plots neurons
        self.axs.scatter(coordinates[:, 0], coordinates[:, 1], zorder=2, s=300)
        self.axs.axis("off")
        self.axs.set_title("Structure of a neural network with the shape {}.".format(self.print_array(self.neurons)))
        self.fig.tight_layout()
        plt.show()

    @staticmethod
    def print_array(arr):
        """ 
        Static method which formats elemtens of arrays consisting of int float or strings for printing.

        Parameters:
            arr: array_like
                Must consist of int, float or strings.
        """
        string = ""
        for i, elem in enumerate(arr):
            if i < len(arr):
                string += str(elem) + ", "
            else:
                string += str(elem)
        return string

def main():
    nn = Node()
    nn.plot_structure()

if __name__ == "__main__":
    main()