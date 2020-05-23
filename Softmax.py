import numpy

import ILayer as interface
from Optimizer import AdamOptimizer
from Optimizer import SGD
from Initializer import Initializer

class Softmax(interface.ILayer):

    def __init__(self, layer_size, input_len, initializer='xavier_uniform', seed=True):
        if seed:
            numpy.random.seed(1000)
        self.prev_layer_size = input_len
        self.layer_size = layer_size
        shape = [input_len, layer_size]
        init = Initializer(input_len, layer_size, shape, initializer)
        self.weights = init.initializeWeights()
        self.biases = numpy.zeros(layer_size)
        self.epsilon = 1e-5
        self.gradients = []
        self.inp = []
        self.totals = []
        self.adam = AdamOptimizer(self.weights)
        self.sgd = SGD(self.weights)

    def forward(self, inputs):
        inputs = numpy.array(inputs)
        self.inp.append(inputs)
        self.input_shape = inputs.shape
        inputs = inputs.flatten()

        totals = numpy.dot(inputs, self.weights) + self.biases
        self.totals.append(totals)
        exp = numpy.exp(totals - numpy.max(totals)) + self.epsilon
        return exp / numpy.sum(exp, axis=0)

    def backward(self, prev_layer):
        result = []
        for i in range(len(prev_layer)):
            result.append(self.backpropSM(prev_layer[i]))
        return result

    def backpropSM(self, d_L_d_out):
        '''
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float
        '''
        # We know only 1 element of d_L_d_out will be nonzero
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            totals = self.totals.pop(0)
            t_exp = numpy.exp(totals - numpy.max(totals))

            # Sum of all e^totals
            S = numpy.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of loss against totals
            d_L_d_t = numpy.array(gradient * d_out_d_t)
            self.gradients.append(d_L_d_t)

            d_L_d_inputs = self.weights @ d_L_d_t
            return d_L_d_inputs.reshape(self.input_shape)

    def updateWeights(self, learn_rate, optimizer):
        gradient = self.gradients
        d_t_d_w = numpy.array(self.inp)
        d_L_d_w = d_t_d_w.T @ gradient
        d_L_d_w = (1/len(self.gradients))*d_L_d_w
        if optimizer == 'Adam':
            # self.adam.alpha = learn_rate
            self.weights = self.adam.backward_pass(d_L_d_w)
        elif optimizer == 'SGD':
            self.sgd.learn_rate = learn_rate
            self.weights = self.sgd.backward_pass(d_L_d_w)
        gradient = numpy.sum(self.gradients, axis=0) / len(self.gradients)
        self.biases -= learn_rate * gradient
        self.gradients = []
        self.inp = []
        self.totals = []
