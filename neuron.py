import random
import numpy as np


class Neuron:
    def __init__(self, prev_layer_size, seed):
        self.potential = 0.0
        self.activation = 0.0
        random.seed(seed*2)
        self.weights = [random.random() - 0.5 if prev_layer_size > 0 else 1 for i in range(prev_layer_size if prev_layer_size > 0 else 1)]

    def __calculate_potential__(self, inputs):
        self.potential = 0.0
        # if isinstance(inputs, list):
        for i in range(len(inputs)):
            self.potential += inputs[i] * self.weights[i]
            # else:
        #     self.potential = inputs
        return self.potential

    def activate(self, inputs, activation_function):
        self.__calculate_potential__(inputs)
        return self.__getattribute__("__activate_"+activation_function+"__")()

    def __activate_sigmoid__(self):
        self.activation = 1.0 / (1 + np.exp(-self.potential))
        return self.activation

    def __activate_relu__(self):
        self.activation = (0 if self.potential < 0 else self.potential)
        return self.activation

    def __activate_tanh__(self):
        self.activation = ((np.exp(self.potential)-np.exp(-self.potential))/(np.exp(self.potential)+np.exp(-self.potential)))
        return self.activation

    def __activate_softmax__(self):
        self.activation = (np.exp(self.potential) / np.sum(np.exp(self.potential), axis=0))
        return self.activation

    def derivative_relu(self):
        return 1 if self.activation > 0 else 0

    def derivative_sigmoid(self):
        return self.activation * (1.0 - self.activation)

    def derivative_tanh(self):
        return 1 - self.activation**2
