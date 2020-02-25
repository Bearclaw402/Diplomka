from neuron import Neuron
import ILayer as interface

class Layer(interface.ILayer):
    def __init__(self, layer_size, prev_layer_size):
        self.outputs = [0] * layer_size
        self.prev_layer_size = prev_layer_size
        self.neurons = [Neuron(prev_layer_size) for i in range(layer_size)]

    def evaluate(self, inputs, activation_function):
        if self.prev_layer_size == 0:
            self.outputs = [self.neurons[i].activate(inputs[i], activation_function) for i in range(len(inputs))]
        else:
            self.outputs = [neuron.activate(inputs, activation_function) for neuron in self.neurons]
        return self.outputs

    def forward(self, prev_layer):
        return self.evaluate(prev_layer, 'relu')

    def backward(self, prev_layer):
        raise NotImplementedError