import layer as l
import ILayer as interface
import numpy

class CNNLayer(interface.ILayer):
    def __init__(self, layer_size, prev_layer_size, activation):
        self.activation = activation
        self.layer = l.Layer(layer_size, prev_layer_size)
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_t = numpy.zeros(self.layer_size)
        self.weights = numpy.zeros([self.layer_size, self.prev_layer_size])
        for neuron in range(len(self.layer.neurons)):
            d_L_d_t[neuron] =  self.layer.neurons[neuron].__getattribute__("derivative_" + self.activation)() * d_L_d_out[neuron]
            self.weights[neuron] = self.layer.neurons[neuron].weights

        # Gradients of totals against weights/biases/input
        d_t_d_w = self.last_input
        d_t_d_w = numpy.array(d_t_d_w)
        d_t_d_b = 1
        d_t_d_inputs = self.weights.T

        # Gradients of loss against weights/biases/input
        s = d_t_d_w[numpy.newaxis].T
        d_L_d_w = d_t_d_w[numpy.newaxis].T @ d_L_d_t[numpy.newaxis]
        d_L_d_b = d_L_d_t * d_t_d_b
        d_L_d_inputs = d_t_d_inputs @ d_L_d_t

        # Update weights / biases
        for neuron in range(len(self.layer.neurons)):
            sdsd = d_L_d_w.T
            self.layer.neurons[neuron].weights -= learn_rate * sdsd[neuron].T

        # self.weights -= learn_rate * d_L_d_w
        # self.biases -= learn_rate * d_L_d_b

        return d_L_d_inputs.reshape(self.last_input_shape)

    def forward(self, prev_layer):
        self.last_input_shape = prev_layer.shape
        prev_layer = prev_layer.flatten()
        self.last_input = prev_layer
        return self.layer.evaluate(prev_layer, self.activation)

    def backward(self, prev_layer, leran_rate):
        return self.backprop(prev_layer, leran_rate)
