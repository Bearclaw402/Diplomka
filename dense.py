import ILayer as interface
import numpy
from Optimizer import AdamOptimizer
from Initializer import Initializer


class Dense(interface.ILayer):
    def __init__(self, layer_size, prev_layer_size, initializer='xavier_uniform', activation='relu'):
        self.activation = activation
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size
        self.activations = []
        self.weights = []
        numpy.random.seed(1000)
        shape = [prev_layer_size, layer_size]
        init = Initializer(prev_layer_size, layer_size, shape, initializer)
        self.weights = init.initializeWeights()
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

    @staticmethod
    def __activate_sigmoid__(potential):
        return 1.0 / (1 + numpy.exp(-potential))

    @staticmethod
    def __activate_relu__(potential):
        potential[potential < 0] = 0
        return potential

    @staticmethod
    def __activate_tanh__(potential):
        return (numpy.exp(potential)-numpy.exp(-potential))/(numpy.exp(potential)+numpy.exp(-potential))

    @staticmethod
    def __activate_cbrt__(potential):
        return numpy.cbrt(potential)

    @staticmethod
    def __activate_comb__(potential):
        result = potential
        index1 = potential >= 0
        index2 = potential < 0
        result[index1] = (numpy.exp(potential[index1])-numpy.exp(-potential[index1]))/(numpy.exp(potential[index1])+numpy.exp(-potential[index1]))
        result[index2] = 0.01*potential[index2]
        return result

    @staticmethod
    def __activate_comb2__(potential):
        result = potential
        index1 = potential >= 0
        index2 = potential < 0
        result[index1] = numpy.sqrt(potential[index1])
        result[index2] = numpy.cbrt(potential[index2])
        return result

    @staticmethod
    def __activate_log__(potential):
        return ((numpy.exp(potential / 2.0) - numpy.exp(-potential / 2.0)) / (numpy.exp(potential / 2.0) + numpy.exp(-potential / 2.0)) / 2.0) + 0.5

    def derivate(self, activation):
        return self.__getattribute__("__derivative_"+self.activation+"__")(activation)

    @staticmethod
    def __derivative_relu__(activation):
        activation[activation > 0] = 1
        activation[activation <= 0] = 0
        return activation

    @staticmethod
    def __derivative_sigmoid__(activation):
        return activation * (1.0 - activation)

    @staticmethod
    def __derivative_tanh__(activation):
        return 1 - activation**2

    @staticmethod
    def __derivative_cbrt__(activation):
        return 1.0/(3*numpy.cbrt(numpy.power(activation, 2)))

    @staticmethod
    def __derivative_comb__(activation):
        result = activation
        index1 = activation >= 0
        index2 = activation < 0
        result[index1] = 1 - activation[index1]**2
        result[index2] = 0.01
        return result

    @staticmethod
    def __derivative_comb2__(activation):
        result = activation
        index1 = activation > 0
        index2 = activation < 0
        result[index1] = 1.0/(2*numpy.sqrt(activation[index1]))
        result[index2] = 1.0/(3*numpy.cbrt(numpy.power(activation[index2], 2)))
        return result

    @staticmethod
    def __derivative_log__(activation):
        return (activation * (1.0-activation))

    def backprop(self, prev_layer):
        result = []
        for i in range(len(prev_layer)):
            d_L_d_t = self.derivate(self.activations.pop(0)) * prev_layer[i]
            d_t_d_inputs = self.weights

            self.gradients.append(d_L_d_t)
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            result.append(d_L_d_inputs.reshape(self.last_input_shape))
        result = numpy.array(result)
        return result

    def forward(self, prev_layer):
        prev_layer = numpy.array(prev_layer)
        self.last_input_shape = prev_layer.shape[1:]
        self.activations = []
        for i in range(prev_layer.shape[0]):
            inp = prev_layer[i].flatten()
            self.last_input.append(inp)
            self.activations.append(self.activate(inp))
        return self.activations

    def backward(self, prev_layer):
        return self.backprop(prev_layer)

    def updateWeights(self, learn_rate):
        gradient = self.gradients
        d_t_d_w = numpy.array(self.last_input)
        d_L_d_w = d_t_d_w.T @ gradient
        d_L_d_w = 1 / (len(self.gradients)) * d_L_d_w
        gradient = numpy.sum(self.gradients, axis=0) / len(self.gradients)
        self.weights = self.adam.backward_pass(d_L_d_w)
        self.biases -= learn_rate * gradient
        self.gradients = []
        self.last_input = []
