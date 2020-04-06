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
        gradient = np.zeros(self.num_outputs)
        gradient[label] = -1 / out[label]

        # Backprop

        for i in range(len(self.layers)):
            layer = self.layers[len(self.layers) - (i + 1)]
            gradient = layer.backward(gradient, lr)

        return loss, acc