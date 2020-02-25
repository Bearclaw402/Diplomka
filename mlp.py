from layer import Layer
from numpy import dot
import json


class Mlp:

    def __init__(self, layer_sizes, learning_rate=0.5, activation_function="sigmoid", epoch_count=10, weights=None):
        self.set_hyperparameters(learning_rate, activation_function, epoch_count)
        self.__init_layers__(layer_sizes)
        if weights is not None:
            self.set_weights(weights)

    def learn(self, inputs, labels):
        for i in range(1, self.epoch_count+1):
            self.__backpropagation__(inputs, labels)

    def __backpropagation__(self, inputs, labels):
        for i in range(len(inputs)):
            out = self.__predict__(inputs[i])
            output_err = self.__calculate_output_error__(out, labels[i])
            weight_err = self.__backpropagate_error__(output_err)
            self.__adapt_synaptic_weights__(weight_err)

    def __calculate_output_error__(self, estimates, labels):
        return [self.layers[-1].neurons[i].__getattribute__("derivative_"+self.activation_function)()*(labels[i] - estimates[i]) for i in range(len(estimates))]

    def __backpropagate_error__(self, output_error):
        err = [[0 for neuron in layer.neurons] for layer in self.layers[:-1]]
        err.append(output_error)
        for m in range(len(self.layers[:-1])-1, 0, -1):
            for i in range(len(self.layers[m].neurons)):
                err[m][i] = self.layers[m].neurons[i].__getattribute__("derivative_"+self.activation_function)()*dot([neuron.weights[i] for neuron in self.layers[m+1].neurons], err[m+1])
        return err

    def __predict__(self, inputs):
        for layer in self.layers:
            outputs = layer.evaluate(inputs, self.activation_function)
            inputs = outputs
        return outputs

    def __adapt_synaptic_weights__(self, error):
        for m in range(1, len(self.layers)):
            for i in range(len(self.layers[m].neurons)):
                for j in range(len(self.layers[m].neurons[i].weights)):
                    self.layers[m].neurons[i].weights[j] += self.learning_rate*error[m][i]*self.layers[m-1].outputs[j]

    def test(self, inputs):
        return [self.__predict__(i) for i in inputs]

    def set_hyperparameters(self, learning_rate, activation_function, epoch_count):
        self.activation_function, self.learning_rate, self.epoch_count = activation_function, learning_rate, epoch_count

    def reset_weights(self):
        self.__init__([len(layer.neurons) for layer in self.layers], self.learning_rate, self.activation_function, self.epoch_count)

    def get_weights(self):
        return [[neuron.weights for neuron in layer.neurons] for layer in self.layers]

    def set_weights(self, weights):
        assert len(self.layers) == len(weights)
        for i in range(len(weights)):
            assert len(self.layers[i].neurons) == len(weights[i])
            for j in range(len(weights[i])):
                assert len(self.layers[i].neurons[j]) == len(weights[i][j])
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                self.layers[i].neurons[j] = weights[i][j]

    def save_model(self, filepath):
        with open(filepath, 'w') as fp:
            json.dump({'learning rate': self.learning_rate, 'activation_function': self.activation_function, 'epoch_count': self.epoch_count, 'weights': self.get_weights()}, fp)

    def load_model(self, filepath):
        with open(filepath, 'r') as fp:
            model = json.load(fp)
            self.__init_layers__([len(layer) for layer in model['weights']])
            self.set_weights(model['weights'])
            self.set_hyperparameters(model['learning_rate'], model['activation_function'], model['epoch_count'])

    def __init_layers__(self, layer_sizes):
        self.layers = [Layer(layer_sizes[j], 0 if j == 0 else layer_sizes[j - 1]) for j in range(len(layer_sizes))]
