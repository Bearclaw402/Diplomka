import numpy as np
import ILayer as interface
from Adam import AdamOptimizer

class Softmax(interface.ILayer):

    def __init__(self, layer_size, input_len):
        np.random.seed(1000)
        self.weights = np.random.randn(input_len, layer_size) / input_len
        self.biases = np.zeros(layer_size)
        self.epsilon = 1e-5
        self.gradients = []
        self.inp = []
        self.totals = []
        # self.thresh = -500
        self.adam = AdamOptimizer(self.weights)

    def forward(self, inputs):
        inputs = np.array(inputs)
        self.inp.append(inputs)
        self.input_shape = inputs.shape
        inputs = inputs.flatten()
        # self.last_input = self.inputs

        totals = np.dot(inputs, self.weights) + self.biases
        # self.last_totals = totals
        self.totals.append(totals)
        # x = totals - np.max(totals)
        # super_threshold_indices = x < self.thresh
        # x[super_threshold_indices] = self.thresh
        exp = np.exp(totals - np.max(totals)) + self.epsilon
        return exp / np.sum(exp, axis=0)

    def backward(self, prev_layer):
        # return self.backpropSM(prev_layer)
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
            # totals = self.totals[0]
            t_exp = np.exp(totals - np.max(totals))

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of loss against totals
            d_L_d_t = np.array(gradient * d_out_d_t)
            self.gradients.append(d_L_d_t)

            d_L_d_inputs = self.weights @ d_L_d_t
            return d_L_d_inputs.reshape(self.input_shape)

    def updateWeights(self, learn_rate):
        gradient = self.gradients
        d_t_d_w = np.array(self.inp)
        d_L_d_w = d_t_d_w.T @ gradient
        d_L_d_w = (1/len(self.gradients))*d_L_d_w
        # self.weights -= learn_rate * d_L_d_w
        # self.adam.alpha = learn_rate
        self.weights = self.adam.backward_pass(d_L_d_w)
        gradient = np.sum(self.gradients, axis=0) / len(self.gradients)
        self.biases -= learn_rate * gradient
        self.gradients = []
        self.inp = []
        self.totals = []
