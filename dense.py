import ILayer as interface
import numpy
import random
from Adam import AdamOptimizer

class Dense(interface.ILayer):
    def __init__(self, layer_size, prev_layer_size, activation):
        self.activation = activation
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size
        self.activations = []
        self.weights = []
        numpy.random.seed(1000)
        self.weights = numpy.random.randn(prev_layer_size, layer_size) / prev_layer_size
        # for i in range(layer_size):
        #     self.weights.append([random.random() - 0.5 for i in
        #                 range(prev_layer_size)])
        # self.weights = numpy.array(numpy.swapaxes(self.weights, 0, 1))
        self.biases = numpy.zeros(layer_size)
        self.last_input = []
        self.gradients = []
        self.adam = AdamOptimizer(self.weights)
        numpy.seterr('raise')

    def __calculate_potential__(self, inputs):
        result = numpy.dot(inputs, self.weights) + self.biases
        return result

    def activate(self, inputs):
        potential = self.__calculate_potential__(inputs)
        return self.__getattribute__("__activate_"+self.activation+"__")(potential)

    def __activate_sigmoid__(self, potential):
        return 1.0 / (1 + numpy.exp(-potential))

    def __activate_relu__(self, potential):
        return 0 if potential < 0 else potential

    def __activate_tanh__(self, potential):
        return (numpy.exp(potential)-numpy.exp(-potential))/(numpy.exp(potential)+numpy.exp(-potential))

    def derivate(self, activation):
        return self.__getattribute__("derivative_"+self.activation)(activation)

    def derivative_relu(self, activation):
        return 1 if activation > 0 else 0

    def derivative_sigmoid(self, activation):
        return activation * (1.0 - activation)

    def derivative_tanh(self, activation):
        return 1 - activation**2


    def backpropFC(self, d_L_d_out):
        # Gradients of totals against weights/biases/input
        d_L_d_t = self.derivate(self.activations.pop(0)) * d_L_d_out
        d_t_d_inputs = self.weights

        self.gradients.append(d_L_d_t)
        d_L_d_inputs = d_t_d_inputs @ d_L_d_t

        return d_L_d_inputs.reshape(self.last_input_shape)

    def forward2(self, prev_layer):
        prev_layer = numpy.array(prev_layer)
        self.last_input_shape = prev_layer.shape[1:]
        self.activations = []
        for i in range(prev_layer.shape[0]):
            inp = prev_layer[i].flatten()
            self.last_input.append(inp)
            self.activations.append(self.activate(inp))
        return self.activations

    def backprop1(self, prev_layer):
        return self.backpropFC(prev_layer)

    def backprop2(self, prev_layer):
        result = []
        for i in range(len(prev_layer)):
            result.append(self.backpropFC(prev_layer[i]))
        result = numpy.array(result)
        return result

    def forward(self, prev_layer):
        return self.forward2(prev_layer)

    def backward(self, prev_layer):
        return self.backprop2(prev_layer)

    def updateWeights(self, learn_rate):
        gradient = self.gradients
        d_t_d_w = numpy.array(self.last_input)
        d_L_d_w = d_t_d_w.T @ gradient
        d_L_d_w = 1 / (len(self.gradients)) * d_L_d_w
        gradient = numpy.sum(self.gradients, axis=0) / len(self.gradients)
        # self.weights -= learn_rate * d_L_d_w
        # self.adam.alpha = learn_rate
        self.weights = self.adam.backward_pass(d_L_d_w)
        self.biases -= learn_rate * gradient
        self.gradients = []
        self.last_input = []
