import layer as l
import ILayer as interface

class CNNLayer(interface.ILayer):
    def __init__(self, layer_size, prev_layer_size, activation, num_filters):
        self.activation = activation
        self.num_filters = num_filters
        self.layers = []
        for filter in range(self.num_filters):
            self.layers.append(l.Layer(layer_size, prev_layer_size))

    def __init__(self, layer_size, prev_layer_size, activation):
        self.activation = activation
        self.layers = []
        for filter in range(self.num_filters):
            self.layers.append(l.Layer(layer_size, prev_layer_size))

    def forward(self, prev_layer):
        for layer in self.layers:
            layer.evaluate(prev_layer, self.activation)

    def backward(self, prev_layer):
        raise NotImplementedError
