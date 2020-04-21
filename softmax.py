import numpy as np
import ILayer as interface

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

    def forward(self, inputs):
        inputs = np.array(inputs)
        self.inp.append(inputs)
        self.last_input_shape = inputs.shape
        self.inputs = inputs.flatten()
        self.last_input = self.inputs

        totals = np.dot(self.inputs, self.weights) + self.biases
        self.last_totals = totals
        self.totals.append(totals)
        # x = totals - np.max(totals)
        # super_threshold_indices = x < self.thresh
        # x[super_threshold_indices] = self.thresh
        exp = np.exp(totals - np.max(totals)) + self.epsilon
        return exp / np.sum(exp, axis=0)

    def backward(self, prev_layer):
        return self.backpropSM(prev_layer)

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

            # e^totals
            # x = self.last_totals - np.max(self.last_totals)
            # super_threshold_indices = x < self.thresh
            # x[super_threshold_indices] = self.thresh
            # t_exp = np.exp(self.last_totals - np.max(self.last_totals)) + self.epsilon
            # t_exp = np.exp(self.last_totals - np.max(self.last_totals))
            totals = self.totals.pop()
            t_exp = np.exp(totals - np.max(totals))

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            # d_t_d_w = self.last_input
            # d_t_d_w = np.array(d_t_d_w)
            # d_t_d_b = 1
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t
            d_L_d_t = np.array(d_L_d_t)

            # self.inp.append(d_t_d_w)
            self.gradients.append(d_L_d_t)

            # Gradients of loss against weights/biases/input
            # s = d_t_d_w[np.newaxis].T
            # d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            # d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # learn_rate = 0.005
            # Update weights / biases
            # self.weights -= learn_rate * d_L_d_w
            # self.biases -= learn_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)

    def updateWeights(self, learn_rate):
        # gradient = np.sum(self.gradients, axis=0) / len(self.gradients)
        # d_t_d_w = np.sum(self.inp, axis=0) / len(self.inp)
        gradient = self.gradients
        d_t_d_w = np.array(self.inp.pop())
        d_t_d_w = d_t_d_w[np.newaxis]
        d_L_d_w = d_t_d_w.T @ gradient
        d_L_d_w = (1/len(self.gradients))*d_L_d_w
        # d_L_d_w = (1/len(self.gradients))*gradient.dot(d_t_d_w.T)
        self.weights -= learn_rate * d_L_d_w
        gradient = np.sum(self.gradients, axis=0) / len(self.gradients)
        self.biases -= learn_rate * gradient
        self.gradients = []
        # self.inp = []
        # self.totals = []
