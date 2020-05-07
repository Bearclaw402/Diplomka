import ILayer as interface
import numpy
from scipy.stats import gennorm
from Adam import AdamOptimizer

class Dense(interface.ILayer):
    def __init__(self, layer_size, prev_layer_size, initializer='xavier_uniform', activation='relu'):
        self.activation = activation
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size
        self.activations = []
        self.weights = []
        numpy.random.seed(1000)
        self.initializeWeights2(type=initializer)
        # self.weights = numpy.random.randn(prev_layer_size, layer_size) / prev_layer_size
        # for i in range(layer_size):
        #     self.weights.append([random.random() - 0.5 for i in
        #                 range(prev_layer_size)])
        # self.weights = numpy.array(numpy.swapaxes(self.weights, 0, 1))
        self.biases = numpy.zeros(layer_size)
        self.last_input = []
        self.gradients = []
        self.adam = AdamOptimizer(self.weights)
        numpy.seterr('raise')

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

    def __calculate_potential__(self, inputs):
        result = numpy.dot(inputs, self.weights) + self.biases
        return result

    def activate(self, inputs):
        potential = self.__calculate_potential__(inputs)
        return self.__getattribute__("__activate_"+self.activation+"__")(potential)

    def __activate_sigmoid__(self, potential):
        return 1.0 / (1 + numpy.exp(-potential))

    def __activate_relu__(self, potential):
        potential[potential < 0] = 0
        return potential
        # return 0 if potential < 0 else potential

    def __activate_tanh__(self, potential):
        return (numpy.exp(potential)-numpy.exp(-potential))/(numpy.exp(potential)+numpy.exp(-potential))

    def __activate_cbrt__(self, potential):
        return numpy.cbrt(potential)

    def __activate_comb__(self, potential):
        result = potential
        index1 = potential >= 0
        index2 = potential < 0
        result[index1] = (numpy.exp(potential[index1])-numpy.exp(-potential[index1]))/(numpy.exp(potential[index1])+numpy.exp(-potential[index1]))
        result[index2] = 0.01*potential[index2]
        return result

    def __activate_comb2__(self, potential):
        result = potential
        index1 = potential >= 0
        index2 = potential < 0
        result[index1] = numpy.sqrt(potential[index1])
        result[index2] = numpy.cbrt(potential[index2])
        return result

    def __activate_log__(self, potential):
        return ((numpy.exp(potential / 2.0) - numpy.exp(-potential / 2.0)) / (numpy.exp(potential / 2.0) + numpy.exp(-potential / 2.0)) / 2.0) + 0.5

    def derivate(self, activation):
        return self.__getattribute__("__derivative_"+self.activation+"__")(activation)

    def __derivative_relu__(self, activation):
        activation[activation > 0] = 1
        activation[activation <= 0] = 0
        return activation
        # return 1 if activation > 0 else 0

    def __derivative_sigmoid__(self, activation):
        return activation * (1.0 - activation)

    def __derivative_tanh__(self, activation):
        return 1 - activation**2

    def __derivative_cbrt__(self, activation):
        return 1.0/(3*numpy.cbrt(numpy.power(activation, 2)))

    def __derivative_comb__(self, activation):
        result = activation
        index1 = activation >= 0
        index2 = activation < 0
        result[index1] = 1 - activation[index1]**2
        result[index2] = 0.01
        return result

    def __derivative_comb2__(self, activation):
        result = activation
        index1 = activation > 0
        index2 = activation < 0
        result[index1] = 1.0/(2*numpy.sqrt(activation[index1]))
        result[index2] = 1.0/(3*numpy.cbrt(numpy.power(activation[index2], 2)))
        return result

    def __derivative_log__(self, activation):
        return (activation * (1.0-activation))

    def backpropFC(self, d_L_d_out):
        # Gradients of totals against weights/biases/input
        d_L_d_t = self.derivate(self.activations.pop(0)) * d_L_d_out
        d_t_d_inputs = self.weights

        self.gradients.append(d_L_d_t)
        d_L_d_inputs = d_t_d_inputs @ d_L_d_t

        return d_L_d_inputs.reshape(self.last_input_shape)

    def forward2(self, prev_layer):
        prev_layer = numpy.array(prev_layer)
        self.last_input_shape = prev_layer.shape[1:]
        self.activations = []
        for i in range(prev_layer.shape[0]):
            inp = prev_layer[i].flatten()
            # inp = prev_layer[i]
            self.last_input.append(inp)
            self.activations.append(self.activate(inp))
        return self.activations

    def backprop1(self, prev_layer):
        return self.backpropFC(prev_layer)

    def backprop2(self, prev_layer):
        result = []
        for i in range(len(prev_layer)):
            result.append(self.backpropFC(prev_layer[i]))
        result = numpy.array(result)
        return result

    def forward(self, prev_layer):
        return self.forward2(prev_layer)

    def backward(self, prev_layer):
        return self.backprop2(prev_layer)

    def updateWeights(self, learn_rate):
        gradient = self.gradients
        d_t_d_w = numpy.array(self.last_input)
        d_L_d_w = d_t_d_w.T @ gradient
        d_L_d_w = 1 / (len(self.gradients)) * d_L_d_w
        gradient = numpy.sum(self.gradients, axis=0) / len(self.gradients)
        # self.weights -= learn_rate * d_L_d_w
        # self.adam.alpha = learn_rate
        self.weights = self.adam.backward_pass(d_L_d_w)
        self.biases -= learn_rate * gradient
        self.gradients = []
        self.last_input = []
