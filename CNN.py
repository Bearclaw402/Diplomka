import numpy

class CNN:

    def __init__(self, num_outputs, loss='categorical_crossentropy'):
        self.layers = []
        self.loss = loss
        self.num_outputs = num_outputs

    def add(self, layer):
        self.layers.append(layer)

    def calculateLoss(self, output, label):
        gradient = numpy.zeros(self.num_outputs)
        loss = 0.0
        # acc = 0.0
        acc = numpy.zeros((2,2))
        if self.loss == 'categorical_crossentropy':
            # acc = 1 if numpy.argmax(output) == label else 0
            acc[label][numpy.argmax(output)] = 1
            loss = -numpy.log(output[label])
            gradient[label] = -1 / output[label]
        elif self.loss == 'bin_loss1':
            acc[label][numpy.argmax(output)] = 1
            p = numpy.zeros(2)
            p[label] = 1.0
            loss = (-p[0]*numpy.log(output[0]))-(p[1]*numpy.log(output[1])/output[1])
            gradient[label] = (p[0]*(-1 / output[0])) + (p[1]*(numpy.log(output[1]) - 1.0) / output[1]**2)
        elif self.loss == 'bin_loss2':
            acc[label][numpy.argmax(output)] = 1
            x = output[label]
            loss = -numpy.log(x) / (1 - (label - x))
            gradient[label] = (-x + (x*numpy.log(x)) + label - 1) / (x*(x-label+1)**2)
        elif self.loss == 'bin_loss3':
            thresh = 0.5
            acc[label][numpy.argmax(output)] = 1
            p = numpy.zeros(2)
            p[label] = 1.0
            x0 = output[0]
            x1 = output[1]
            loss = (-p[0]*numpy.log(x0))-(p[1]*numpy.log(x1)/(x1+thresh)**2)
            gradient[label] = (p[0]*(-1 / x0)) + (p[1]*2.0*((-thresh*x1) + (x1 * numpy.log(x1)) - (thresh/2.0)) / (x1 * (x1 + thresh)**3))
        elif self.loss == 'bin_loss4':
            thresh0 = 0.3
            thresh1 = 0.5-thresh0
            acc[label][numpy.argmax(output)] = 1
            x = output[label]
            loss = -numpy.log(x) / (x + (thresh0 + thresh1 * label))**2
            gradient[label] = 2 * ((-0.5 * x) + (x*numpy.log(x)) - (0.1*label) - 0.15) / (x*(x+(thresh1 * label)+thresh0)**3)
        return gradient, loss, acc

    def forwardBatch(self, batch, labels):
        output = ((batch / 255) - 0.5)
        # output = (batch / 255)
        for i in range(0, len(self.layers) - 1):
            output = self.layers[i].forward(output)
        total_loss = 0.0
        total_acc = numpy.zeros((2,2))
        gradients = []
        for i in range(len(batch)):
            out = self.layers[- 1].forward(output[i])
            gradient, loss, acc = self.calculateLoss(out,labels[i])
            total_acc += acc
            total_loss += loss
            gradients.append(gradient)
        loss = total_loss / len(batch)
        # acc = total_acc / len(batch)
        return gradients, loss, total_acc

    def train3(self, im, label, lr=.01):
        gradients, loss, acc = self.forwardBatch(im, label)

        for i in range(len(self.layers)- 1, 0, -1):
            layer = self.layers[i]
            gradients = layer.backward(gradients)
            layer.updateWeights(lr)

        return loss, acc