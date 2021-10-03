import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from neural_network import *


class Multilayer_perceptron(NeuralNetwork):
    """ 
    Multilayer network implementation of perceptron algorithm. Child which inherits from neural network base class.

    Parameters:
        Xtr: array_like
            Training data structured as an matrix with N-rows corresponing to datapoint and l-columns representing features.
        Ytr: array_like
            Training data labels. Structured as l-columns with 1 row. Values contained should have class belonging information.
        network_structure: array_like
            Array with information of the structure of the network. len(structure) represents the number of layers, whilst values represents neurons within.
    """
    def __init__(self, Xtr, Ytr, network_structure):
        super().__init__(Xtr, Ytr, network_structure)
        self.prev_delta_w = []

    def forward_propagate(self, Xtr, Ytr):

        y_prev = np.atleast_2d(Xtr)
        self.v_arr = []
        self.y_arr = [Xtr]
        
        # loops through all layers which has a weight defined.
        for r, layer in enumerate(self.network.layers[1:]):
            v = y_prev @ layer.w_mat.T

            # add column of ones to v-matrix
            if layer.index < len(layer.network.shape) - 1:
                v = np.append(v, np.ones((len(v), 1)), axis=1)

            self.v_arr.append(v)
            self.y = self.sigmoid(v)
            self.y_arr.append(self.y)

            y_prev = self.y

        y_hat = self.y

        self.error = (self.y.squeeze() - Ytr).reshape(len(self.y), 1)

        return np.round(y_hat)

    def backward_propagate(self, error):
        last_delta = error * self.sigmoid_derivative(self.v_arr[-1]) #[:, None]

        deltas = [last_delta]

        # iterates backwards through layers but does not count over input layer
        for layer in self.network.layers[:0:-1]:
            r = layer.index
            if r == 1:
                break

            # snips away bias in w_matrix
            err = deltas[-1] @ layer.w_mat[:, :-1]

            deltas_in_layer = []

            if r > 1:
                # By indexing we clip away the last element as it represents the bias which is differentiated away.
                der = self.sigmoid_derivative(self.v_arr[r - 2][:, :-1])

                # Calculate delta from elementwise multiplication with error with sigmoid derivative
                for error_node, der_node in zip(err.T, der.T):
                    
                    # For each node calculate deltas
                    delta_node = error_node * der_node
                    deltas_in_layer.append(delta_node)

                deltas.append(np.array(deltas_in_layer).T)

        self.update_weights(deltas)
        
    def update_weights(self, deltas):
        # loop through each layer except input layer.
        for r, layer in enumerate(self.network.layers[:0:-1]):
            # opposite_index = len(self.network.layers[:0:-1]) - r

            if layer.is_output:
                delta_w = - self.learning_rate * deltas[r].T @ self.y_arr[layer.index - 1]

            else:
                delta_w = []
                for i, delta in enumerate(deltas[r].T):
                    
                    delta_w_node = - self.learning_rate * delta[:, None].T @ self.y_arr[layer.index - 1]
                    
                    delta_w.append(delta_w_node)

            for i, neuron in enumerate(layer.nodes):
                if layer.is_output:
                    neuron.change_class_weights(neuron.w + delta_w, self.alpha)
                else:
                    neuron.change_class_weights(neuron.w + delta_w[i], self.alpha)


    def train(self, mu=1, epochs=1000, alpha=0, ax=None):
        self.alpha = alpha

        self.learning_rate = mu
        epochs = np.arange(epochs)

        errors_arr = []
        for epoch in epochs:
            y_hat = self.forward_propagate(self.Xtr, self.Ytr)
            
            self.backward_propagate(self.error)
            
            self.Ytr = np.atleast_2d(self.Ytr)

            predictions = y_hat.T != self.Ytr
            errors = len(predictions[predictions])

            errors_arr.append(errors)

        ax.plot(epochs, errors_arr)
        ax.set_title("Errors")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Errors")

    def test(self, Xte, Yte):
        y_hat = self.forward_propagate(Xte, Yte)
        return y_hat

    def plot_training(self, ax, lr, time):

        x1_range = np.linspace(np.min(self.Xtr[:, 0]), np.max(self.Xtr[:, 0]), 50)
        x2_range = np.linspace(np.min(self.Xtr[:, 1]), np.max(self.Xtr[:, 1]), 50)

        xx, yy = np.meshgrid(x1_range, x2_range)
        
        xx, yy = xx.reshape(len(x1_range)**2), yy.reshape(len(x2_range)**2)
        inp = np.transpose(np.vstack((xx, yy)))
        inp = np.c_[inp, np.ones(inp.shape[0])]
        y = np.atleast_2d(np.zeros(len(inp)))

        y_hat = self.test(inp, y)
        y_hat = y_hat.reshape(len(x1_range), len(x1_range))

        # print(np.round(y_hat)[0:10, 0:10])

        ax.contourf(x1_range,x2_range, y_hat, cmap="cool")
        ax.scatter(self.Xtr[:, 0], self.Xtr[:, 1], c=list(self.Ytr), s=5)
        ax.set_title("lr = {:.4f}, t = {:.3f}".format(lr, time))
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

