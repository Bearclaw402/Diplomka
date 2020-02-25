import layer as l
import softmax as sm
import numpy as np

class CNN:

    def __init__(self, num_outputs):
        self.layers = []
        self.num_outputs = num_outputs

    def add(self, layer):
        self.layers.append(layer)

    def forwardImage(self, image, label):
        output = ((image / 255) - 0.5)
        for layer in self.layers:
            output = layer.forward(output)
        loss = -np.log(output[label])
        acc = 1 if np.argmax(output) == label else 0
        return output, loss, acc

    # def train(self):
    #     gradient = np.zeros(self.num_outputs)
    #     gradient = self.softmax.backward(gradient)
    #     for layer in self.layers:
    #         gradient = layer.backward(gradient)

    def train(self, im, label, lr=.005):
        '''
        Completes a full training step on the given image and label.
        Returns the cross-entropy loss and accuracy.
        - image is a 2d numpy array
        - label is a digit
        - lr is the learning rate
        '''
        # Forward
        out, loss, acc = self.forwardImage(im, label)

        # Calculate initial gradient
        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        # Backprop

        layer = self.layers[len(self.layers) - 1]
        gradient = layer.backprop(gradient, lr)
        layer = self.layers[len(self.layers) - 2]
        gradient = layer.backward(gradient)
        layer = self.layers[len(self.layers) - 3]
        gradient = layer.backprop(gradient, lr)
        # TODO: backprop MaxPool2 layer
        # TODO: backprop Conv3x3 layer

        return loss, acc

    def flatten(self, input):
        output = []
        for filter in range(len(input)):
            for row in range(len(input[0])):
                for column in range(len(input[0][0])):
                    #for i in range(len(input[0])):
                        output.append(input[filter][row][column])
        return output