import layer as l
import ILayer as interface
import numpy

class CNNLayer(interface.ILayer):
    def __init__(self, layer_size, prev_layer_size, activation):
        self.activation = activation
        self.layer = l.Layer(layer_size, prev_layer_size)
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size
        self.last_input = []
        self.gradients = []
        self.inp = []

    def backpropFC(self, d_L_d_out):
        d_L_d_t = numpy.zeros(self.layer_size)
        self.weights = numpy.zeros([self.layer_size, self.prev_layer_size])
        for neuron in range(len(self.layer.neurons)):
            d_L_d_t[neuron] =  self.layer.neurons[neuron].__getattribute__("derivative_" + self.activation)() * d_L_d_out[neuron]
            self.weights[neuron] = self.layer.neurons[neuron].weights

        # Gradients of totals against weights/biases/input
        d_t_d_w = self.last_input.pop()
        d_t_d_w = numpy.array(d_t_d_w)
        # d_t_d_b = 1
        d_t_d_inputs = self.weights.T

        self.inp.append(d_t_d_w)
        self.gradients.append(d_L_d_t)

        # Gradients of loss against weights/biases/input
        # d_L_d_w = d_t_d_w[numpy.newaxis].T @ d_L_d_t[numpy.newaxis]
        # d_L_d_b = d_L_d_t * d_t_d_b
        d_L_d_inputs = d_t_d_inputs @ d_L_d_t

        # learn_rate = 0.005
        # Update weights / biases
        # for neuron in range(len(self.layer.neurons)):
        #     sdsd = d_L_d_w.T
        #     self.layer.neurons[neuron].weights -= learn_rate * sdsd[neuron].T
        # self.biases -= learn_rate * d_L_d_b

        return d_L_d_inputs.reshape(self.last_input_shape)

    def forward1(self, prev_layer):
        prev_layer = numpy.array(prev_layer)
        self.last_input_shape = prev_layer.shape
        prev_layer = prev_layer.flatten()
        self.last_input.append(prev_layer)
        return self.layer.evaluate(prev_layer, self.activation)

    def forward2(self, prev_layer):
        prev_layer = numpy.array(prev_layer)
        self.last_input_shape = prev_layer.shape[1:]
        outs = []
        for i in range(prev_layer.shape[0]):
            inp = prev_layer[i].flatten()
            self.last_input.append(inp)
            outs.append(self.layer.evaluate(inp, self.activation))
        # outs = outs[0]
        return outs

    def backprop1(self, prev_layer):
        return self.backpropFC(prev_layer)

    def backprop2(self, prev_layer):
        result = []
        for i in range(len(prev_layer)):
            result.append(self.backpropFC(prev_layer[i]))
        result = numpy.array(result)
        return result

    def forward(self, prev_layer):
        if (prev_layer.ndim > 3):
            return self.forward2(prev_layer)
        else:
            return self.forward1(prev_layer)

    def backward(self, prev_layer):
        return self.backprop2(prev_layer)

    def updateWeights(self, learn_rate):
        # gradient = numpy.sum(self.gradients, axis=0) / len(self.gradients)
        # d_t_d_w = numpy.sum(self.inp, axis=0) / len(self.inp)
        gradient = self.gradients
        d_t_d_w = numpy.array(self.inp)
        d_L_d_w = d_t_d_w.T @ gradient
        for neuron in range(len(self.layer.neurons)):
            sdsd = 1/(len(self.gradients)) * d_L_d_w.T
            self.layer.neurons[neuron].weights -= learn_rate * sdsd[neuron].T
        self.gradients = []
        self.inp = []
