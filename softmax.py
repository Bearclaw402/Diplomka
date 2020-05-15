import numpy
from scipy.stats import gennorm

import ILayer as interface
from Optimizer import AdamOptimizer

class Softmax(interface.ILayer):

    def __init__(self, layer_size, input_len, initializer='xavier_uniform'):
        numpy.random.seed(1000)
        self.prev_layer_size = input_len
        self.layer_size = layer_size
        self.initializeWeights2(initializer)
        self.biases = numpy.zeros(layer_size)
        self.epsilon = 1e-5
        self.gradients = []
        self.inp = []
        self.totals = []
        self.adam = AdamOptimizer(self.weights)

    def initializeWeights(self, type='xavier_uniform'):
        n_in = self.prev_layer_size
        n_out = self.layer_size
        shape = [self.prev_layer_size, self.layer_size]
        if type == 'xavier_normal':
            scale = numpy.sqrt(2. / (n_in + n_out))
            self.weights = numpy.random.normal(loc=0.0, scale=scale, size=shape)
        elif type == 'he_normal':
            scale = numpy.sqrt(2. / n_in)
            self.weights = numpy.random.normal(loc=0.0, scale=scale, size=shape)
        elif type == 'xavier_uniform':
            scale = numpy.sqrt(6. / (n_in + n_out))
            self.weights = numpy.random.uniform(low=-scale, high=scale, size = shape)
        elif type == 'he_uniform':
            scale = numpy.sqrt(6. / n_in)
            self.weights = numpy.random.uniform(low=-scale, high=scale, size = shape)
        elif type == 'random_uniform':
            scale = 0.05
            self.weights = numpy.random.uniform(low=-scale, high=scale, size = shape)
        elif type == 'random_normal':
            scale = 0.05
            self.weights = numpy.random.normal(loc=0.0, scale=scale, size=shape)
        else:
            scale = 1. / self.prev_layer_size
            self.weights = numpy.random.normal(loc=0.0, scale=scale, size=shape)

    def initializeWeights2(self, type='xavier_uniform'):
        initializer = type.split('_')[0]
        distribution = type.split('_')[1]
        n_in = self.prev_layer_size
        n_out = self.layer_size
        shape = [self.prev_layer_size, self.layer_size]
        mean = 0.0
        dev = 0.0
        low = 0.0
        high = 0.0
        df = 1.0
        beta = 1.0
        if initializer == 'xavier':
            if 'norm' in distribution:
                dev = numpy.sqrt(2.0 / (n_in + n_out))
            elif distribution == 'uniform':
                low = -numpy.sqrt(6.0 / (n_in + n_out))
                high = numpy.sqrt(6.0 / (n_in + n_out))
            else:
                dev = numpy.sqrt(1.0 / (n_in + n_out))
        elif initializer == 'he':
            if 'norm' in distribution:
                dev = numpy.sqrt(2.0 / n_in)
            elif distribution == 'uniform':
                low = -numpy.sqrt(6.0 / n_in)
                high = numpy.sqrt(6.0 / n_in)
            else:
                dev = numpy.sqrt(1.0 / n_in)
        elif initializer == 'random':
            low = -0.05
            high = 0.05
        else:
            dev = 1. / n_in

        if distribution == 'normal':
            self.weights = self.normal(shape,mean,dev)
        elif distribution == 'uniform':
            self.weights = self.uniform(shape,low,high)
        elif distribution == 'student':
            self.weights = dev*self.student(shape,df)
        elif distribution == 'chisqr':
            self.weights = dev*self.chisqr(shape,df)
        elif distribution == 'gennorm':
            self.weights = self.gennorm(shape,beta,mean,dev)
        elif distribution == 'lognorm':
            self.weights = self.lognorm(shape,mean,dev)
        else:
            self.weights = self.normal(shape)

    def normal(self, shape, mean=0.0, dev=0.05):
        return numpy.random.normal(loc=mean, scale=dev, size=shape)

    def uniform(self, shape, low=-0.05, high=0.05):
        return numpy.random.uniform(low=low, high=high, size=shape)

    def student(self, shape, df=1.0):
        return numpy.random.standard_t(df=df, size=shape)

    def chisqr(self, shape, df=1.0):
        return numpy.random.chisquare(df=df, size=shape)

    def gennorm(self, shape, beta=1.0, mean=0.0, dev=0.05):
        return gennorm.rvs(beta=beta, loc=mean, scale=dev, size=shape)

    def lognorm(self, shape, mean=0.0, dev=0.05):
        return numpy.random.lognormal(mean=mean, sigma=dev, size=shape)

    def forward(self, inputs):
        inputs = numpy.array(inputs)
        self.inp.append(inputs)
        self.input_shape = inputs.shape
        inputs = inputs.flatten()

        totals = numpy.dot(inputs, self.weights) + self.biases
        self.totals.append(totals)
        exp = numpy.exp(totals - numpy.max(totals)) + self.epsilon
        return exp / numpy.sum(exp, axis=0)

    def backward(self, prev_layer):
        result = []
        for i in range(len(prev_layer)):
            result.append(self.backpropSM(prev_layer[i]))
        return result

    def backpropSM(self, d_L_d_out):
        '''
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float
        '''
        # We know only 1 element of d_L_d_out will be nonzero
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            totals = self.totals.pop(0)
            t_exp = numpy.exp(totals - numpy.max(totals))

            # Sum of all e^totals
            S = numpy.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of loss against totals
            d_L_d_t = numpy.array(gradient * d_out_d_t)
            self.gradients.append(d_L_d_t)

            d_L_d_inputs = self.weights @ d_L_d_t
            return d_L_d_inputs.reshape(self.input_shape)

    def updateWeights(self, learn_rate):
        gradient = self.gradients
        d_t_d_w = numpy.array(self.inp)
        d_L_d_w = d_t_d_w.T @ gradient
        d_L_d_w = (1/len(self.gradients))*d_L_d_w
        self.weights = self.adam.backward_pass(d_L_d_w)
        gradient = numpy.sum(self.gradients, axis=0) / len(self.gradients)
        self.biases -= learn_rate * gradient
        self.gradients = []
        self.inp = []
        self.totals = []
