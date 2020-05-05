import numpy as np

class CNN:

    def __init__(self, num_outputs):
        self.layers = []
        self.num_outputs = num_outputs

    def add(self, layer):
        self.layers.append(layer)

    def forwardBatch(self, batch, labels):
        output = ((batch / 255) - 0.5)
        # output = (batch / 255)
        for i in range(0, len(self.layers) - 1):
            output = self.layers[i].forward(output)
        total_loss = 0.0
        total_acc = 0.0
        gradients = []
        for i in range(len(batch)):
            out = self.layers[- 1].forward(output[i])
            loss = -np.log(out[labels[i]])
            acc = 1 if np.argmax(out) == labels[i] else 0
            total_acc += acc
            total_loss += loss
            gradient = np.zeros(self.num_outputs)
            gradient[labels[i]] = -1 / out[labels[i]]
            gradients.append(gradient)
        loss = total_loss / len(batch)
        acc = total_acc / len(batch)
        return gradients, loss, acc

    def train3(self, im, label, lr=.01):
        gradients, loss, acc = self.forwardBatch(im, label)

        for i in range(len(self.layers)- 1, 0, -1):
            layer = self.layers[i]
            gradients = layer.backward(gradients)
            layer.updateWeights(lr)

        return loss, acc