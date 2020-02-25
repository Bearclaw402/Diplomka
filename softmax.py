import numpy as np
import ILayer as interface

class Softmax(interface.ILayer):

    def __init__(self, layer_size, input_len):
        self.weights = np.random.randn(input_len, layer_size) / input_len
        self.biases = np.zeros(layer_size)

    def forward(self, inputs):
        inputs = np.array(inputs)
        self.last_input_shape = inputs.shape
        self.inputs = self.flatten(inputs)
        self.last_input = self.inputs

        #input_len, nodes = self.weights.shape

        totals = np.dot(self.inputs, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def flatten(self, input):
        output = []
        for filter in range(len(input)):
            for row in range(len(input[0])):
                for column in range(len(input[0][0])):
                    #for i in range(len(input[0])):
                        output.append(input[filter][row][column])
        return output

    def backprop(self, d_L_d_out, learn_rate):
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
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t
            d_t_d_w = np.array(d_t_d_w)
            d_L_d_t = np.array(d_L_d_t)
            # Gradients of loss against weights/biases/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)