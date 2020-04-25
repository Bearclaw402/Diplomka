import numpy as np

class CNN:

    def __init__(self, num_outputs):
        self.layers = []
        self.num_outputs = num_outputs

    def add(self, layer):
        self.layers.append(layer)

    def forwardImage(self, image, label):
        output = ((image / 255) - 0.5)
        # i = 0
        for layer in self.layers:
            # i += 1
            # if i >=len(self.layers):
            #     output = layer.forward(output[0])
                # output = output[0]
            # else:
            output = layer.forward(output)
        loss = -np.log(output[label])
        acc = 1 if np.argmax(output) == label else 0
        # acc = 1 if np.argmax(output) == label[0] else 0
        return output, loss, acc

    def forwardBatch(self, batch, labels):
        output = ((batch / 255) - 0.5)
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
        return gradients, total_loss, total_acc

    def train(self, im, label, lr=.01):
        '''
        Completes a full training step on the given image and label.
        Returns the cross-entropy loss and accuracy.
        - image is a 2d numpy array
        - label is a digit
        - lr is the learning rate
        '''
        batch_size = im.shape[0]
        avg_loss = 0.0
        avg_acc = 0.0
        for i in range(batch_size):
            # Forward
            out, loss, acc = self.forwardImage(im[i], label[i])

            avg_loss += loss
            avg_acc += acc

            # Calculate initial gradient
            gradient = np.zeros(self.num_outputs)
            gradient[label[i]] = -1 / out[label[i]]

            # Backprop

            for i in range(len(self.layers)- 1, 0, -1):
                layer = self.layers[i]
                gradient = layer.backward(gradient)

        for i in range(len(self.layers)- 1, 0, -1):
            layer = self.layers[i]
            layer.updateWeights(lr * batch_size)
        return avg_loss, avg_acc

    def train2(self, im, label, lr=.01):
        batch_size = im.shape[0]
        gradients = []
        avg_loss = 0.0
        avg_acc = 0.0
        for i in range(batch_size):
            # Forward
            out, loss, acc = self.forwardImage(im[i], label[i])
            avg_loss += loss
            avg_acc += acc

            # Calculate initial gradient
            gradient = np.zeros(self.num_outputs)
            gradient[label[i]] = -1 / out[label[i]]

            # Backprop
            gradients.append(self.layers[-1].backward(gradient))
            self.layers[-1].updateWeights(lr * batch_size)

        for i in range(len(self.layers)- 2, 0, -1):
            layer = self.layers[i]
            gradients = layer.backward(gradients)
            layer.updateWeights(lr * batch_size)

        return avg_loss, avg_acc

    def train3(self, im, label, lr=.01):
        batch_size = im.shape[0]
        gradients, loss, acc = self.forwardBatch(im, label)

        for i in range(len(self.layers)- 1, 0, -1):
            layer = self.layers[i]
            gradients = layer.backward(gradients)
            layer.updateWeights(lr)

        return loss, acc