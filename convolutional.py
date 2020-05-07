from scipy.stats import gennorm

import ILayer as interface
from scipy import ndimage
from scipy import signal
import numpy
from dense import Dense
from Adam import AdamOptimizer


class Conv(interface.ILayer):
    def __init__(self, prev_layer_size, num_filters, filter_size, stride = 1, padding = 0, initializer='xavier_uniform', activation=None):
        numpy.random.seed(10+prev_layer_size*num_filters)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias = numpy.random.randint(1, size=(num_filters, prev_layer_size))
        self.activation = activation
        # self.filters = numpy.random.randn(num_filters, prev_layer_size, filter_size, filter_size) / (filter_size * filter_size)
        self.initializeWeights2(prev_layer_size, type=initializer)
        self.gradients = []
        self.inp = []
        self.output = []
        self.adam = AdamOptimizer(self.filters)

    def initializeWeights(self, prev_layer_size, type='xavier_uniform'):
        n_in = self.filter_size * self.filter_size * prev_layer_size
        n_out = self.filter_size * self.filter_size * self.num_filters
        shape = [self.num_filters, prev_layer_size, self.filter_size, self.filter_size]

        if type == 'xavier_normal':
            scale = numpy.sqrt(2. / (n_in + n_out))
            self.filters = numpy.random.normal(loc=0.0, scale=scale, size=shape)
        elif type == 'he_normal':
            scale = numpy.sqrt(2. / n_in)
            self.filters = numpy.random.normal(loc=0.0, scale=scale, size=shape)
        elif type == 'xavier_uniform':
            scale = numpy.sqrt(6. / (n_in + n_out))
            self.filters = numpy.random.uniform(low=-scale, high=scale, size = shape)
        elif type == 'he_uniform':
            scale = numpy.sqrt(6. / n_in)
            self.filters = numpy.random.uniform(low=-scale, high=scale, size = shape)
        elif type == 'random_uniform':
            scale = 0.05
            self.filters = numpy.random.uniform(low=-scale, high=scale, size = shape)
        elif type == 'random_normal':
            scale = 0.05
            self.filters = numpy.random.normal(loc=0.0, scale=scale, size=shape)
        else:
            scale = 1. / (self.filter_size * self.filter_size)
            self.filters = numpy.random.normal(loc=0.0, scale=scale, size=shape)

    def initializeWeights2(self, prev_layer_size, type='xavier_uniform'):
        initializer = type.split('_')[0]
        distribution = type.split('_')[1]
        n_in = self.filter_size * self.filter_size * prev_layer_size
        n_out = self.filter_size * self.filter_size * self.num_filters
        shape = [self.num_filters, prev_layer_size, self.filter_size, self.filter_size]
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
            self.filters = self.normal(shape,mean,dev)
        elif distribution == 'uniform':
            self.filters = self.uniform(shape,low,high)
        elif distribution == 'student':
            self.filters = dev*self.student(shape,df)
        elif distribution == 'chisqr':
            self.filters = dev*self.chisqr(shape,df)
        elif distribution == 'gennorm':
            self.filters = self.gennorm(shape,beta,mean,dev)
        elif distribution == 'lognorm':
            self.filters = self.lognorm(shape,mean,dev)
        else:
            self.filters = self.normal(shape)

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

    def activate(self, output):
        self.output = self.__getattribute__("__activate_"+self.activation+"__")(output)
        return self.output

    def __activate_relu__(self, output):
        output[output < 0] = 0
        return output

    def __activate_tanh__(self, output):
        return (numpy.exp(output) - numpy.exp(-output)) / (numpy.exp(output) + numpy.exp(-output))

    def __activate_sigmoid__(self, output):
        return 1.0 / (1 + numpy.exp(-output))

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


    def derivate(self):
        return self.__getattribute__("__derivative_"+self.activation+"__")(self.output)

    def __derivative_relu__(self, activation):
        activation[activation > 0] = 1
        activation[activation <= 0] = 0
        return activation

    def __derivative_tanh__(self, activation):
        return 1 - activation**2

    def __derivative_sigmoid__(self, activation):
        return activation * (1.0 - activation)

    def __derivative_cbrt__(self, activation):
        return 1.0/(3*numpy.cbrt(numpy.power(activation, 2)))

    def __derivative_comb__(self, activation):
        result = activation
        index1 = activation >= 0
        index2 = activation < 0
        result[index1] = 1 - activation[index1] ** 2
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

    def conv2_batch(self, input):
        m = input.shape[0]
        if input.ndim <= 3:
            input = input[:, numpy.newaxis, :, :]
            # input = numpy.swapaxes(input, 0, 1)
        self.depth = input.shape[1]
        self.width = (input.shape[2] + 2 * self.padding - self.filter_size + self.stride) // self.stride
        self.height = (input.shape[3] + 2 * self.padding - self.filter_size + self.stride) // self.stride

        if self.padding > 0:
            tmp = numpy.zeros([input.shape[0], input.shape[1], input.shape[2] + self.padding * 2, input.shape[3] + self.padding * 2])
            tmp[:, :, self.padding:tmp.shape[2] - self.padding, self.padding:tmp.shape[3] - self.padding] = input
            self.input = tmp
        else:
            self.input = input

        y = numpy.zeros([m, self.num_filters, self.width, self.height])

        fils = numpy.repeat(self.filters[numpy.newaxis, :, :, :, :], m, axis=0)
        fils = numpy.flip(fils, 4)
        fils = numpy.flip(fils, 3)

        f = self.input.shape[2] - self.width
        s = False
        if f % 2 == 1:
            s = True
        for k in range(self.num_filters):  # each of the filters
            for d in range(self.depth):  # each slice of the input
                if s:
                    y[:, k, :, :] += ndimage.convolve(self.input[:, d, :, :], fils[:, k, d], mode='constant', cval=0.0)[:, (f // 2):-(f // 2) - 1, (f // 2):-(f // 2) - 1]
                else:
                    y[:, k, :, :] += ndimage.convolve(self.input[:, d, :, :], fils[:, k, d], mode='constant', cval=0.0)[:, f//2:-(f//2),f//2:-(f//2)]

        return y

    def convolve2(self, input):
        m = input.shape[0]
        if input.ndim <= 3:
            input = input[:, numpy.newaxis, :, :]
            # input = numpy.swapaxes(input, 0, 1)
        self.depth = input.shape[1]
        self.width = (input.shape[2] + 2 * self.padding - self.filter_size + self.stride) // self.stride
        self.height = (input.shape[3] + 2 * self.padding - self.filter_size + self.stride) // self.stride

        if self.padding > 0:
            tmp = numpy.zeros([input.shape[0], input.shape[1], input.shape[2] + self.padding * 2, input.shape[3] + self.padding * 2])
            tmp[:, :, self.padding:tmp.shape[2] - self.padding, self.padding:tmp.shape[3] - self.padding] = input
            self.input = tmp
        else:
            self.input = input

        output = numpy.zeros([m, self.num_filters, self.width, self.height])

        fils = numpy.repeat(self.filters[numpy.newaxis, :, :, :, :], m, axis=0)
        # fils = numpy.flip(fils, 3)
        # fils = numpy.flip(fils, 4)

        for row in range(self.height):
            for column in range(self.width):
                rows = self.input[:, :, row * self.stride: row * self.stride + self.filter_size,
                           column * self.stride:column * self.stride + self.filter_size]
                for filter in range(self.num_filters):
                    ss = rows * fils[:, filter, :]
                    dd = numpy.sum(ss, axis=(1, 2, 3))
                    output[:, filter, row, column] = dd
        return output

    def backpropC4(self, d_L_d_out):
        m = d_L_d_out.shape[0]
        # d_out_pad = numpy.zeros([d_L_d_out.shape[0], d_L_d_out.shape[1], self.input.shape[1], self.input.shape[2]])
        d_out_pad = numpy.zeros([d_L_d_out.shape[0], d_L_d_out.shape[1], self.input.shape[2], self.input.shape[3]])
        pad = (d_out_pad.shape[2] - d_L_d_out.shape[2]) // 2
        if (d_out_pad.shape[2] - d_L_d_out.shape[2]) % 2 == 1:
            d_out_pad[:, :, pad:d_out_pad.shape[2] - pad - 1, pad:d_out_pad.shape[3] - pad - 1] = d_L_d_out
        else:
            d_out_pad[:, :, pad:d_out_pad.shape[2] - pad, pad:d_out_pad.shape[3] - pad] = d_L_d_out

        # inps = numpy.array(self.inp)
        inps = numpy.array(self.input)
        d_L_d_input = numpy.zeros(inps.shape)
        fils = numpy.repeat(self.filters[numpy.newaxis, :, :, :, :], m, axis=0)

        for filter in range(self.num_filters):
            for d in range(self.depth):
                    d_L_d_input[:, d, :, :] += ndimage.convolve(d_out_pad[:, filter, :, :], fils[:, filter, d])
        d_L_d_input = d_L_d_input / m
        return d_L_d_input

    def backpropC5(self, d_L_d_out):
        m = d_L_d_out.shape[0]
        # d_out_pad = numpy.zeros([d_L_d_out.shape[0], d_L_d_out.shape[1], self.input.shape[1], self.input.shape[2]])
        d_out_pad = numpy.zeros([d_L_d_out.shape[0], d_L_d_out.shape[1], self.input.shape[2], self.input.shape[3]])
        pad = (d_out_pad.shape[2] - d_L_d_out.shape[2]) // 2
        if (d_out_pad.shape[2] - d_L_d_out.shape[2]) % 2 == 1:
            d_out_pad[:, :, pad:d_out_pad.shape[2] - pad - 1, pad:d_out_pad.shape[3] - pad - 1] = d_L_d_out
        else:
            d_out_pad[:, :, pad:d_out_pad.shape[2] - pad, pad:d_out_pad.shape[3] - pad] = d_L_d_out

        inps = numpy.array(self.inp)
        # inps = numpy.array(self.input)
        d_L_d_input = numpy.zeros(inps.shape)
        fils = numpy.repeat(self.filters[numpy.newaxis, :, :, :, :], m, axis=0)
        #
        for row in range(self.height):
            for column in range(self.width):
                rows_i = d_out_pad[:, :, row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size]
                for d in range(self.depth):
                    ss = rows_i * fils[:, :, d]
                    dd = numpy.sum(ss, axis=(1,2,3))
                    d_L_d_input[:, d, row, column] += dd

        d_L_d_input = d_L_d_input / m
        return d_L_d_input

    def filterSet(self, d_L_d_out, learn_rate):
        x = numpy.moveaxis(self.input, 1, -1)
        x_padded_bcast = numpy.expand_dims(x, axis=-1)  # shape = (w, h, d, 1)
        dz = numpy.moveaxis(d_L_d_out, 1, -1)
        dZ_bcast = numpy.expand_dims(dz, axis=-2)  # shape = (i, j, 1, k)
        d_L_d_filters = numpy.zeros(self.filters.shape)
        for a in range(self.filter_size):
            for b in range(self.filter_size):
                asd = 1/len(self.gradients) * numpy.sum(
                    dZ_bcast * x_padded_bcast[:, a:a + self.width,
                               b:b + self.height, :, :], axis=(0, 1,2))
                d_L_d_filters[:, :, a, b] = asd.swapaxes(0,1)

        self.filters = self.adam.backward_pass(d_L_d_filters)

    def forward(self, prev_layer):
        # if len(prev_layer[0]) > 1 and self.stride == 1:
        #     result = self.conv2_batch(prev_layer)
        # else:
        result = self.convolve2(prev_layer)
        self.inp = self.input
        if self.activation != None:
            result = self.activate(result)
        return result
        # if (prev_layer.ndim > 3):
        #     return self.forward2(prev_layer)
        # else:
        #     return self.forward1(prev_layer)

    def backward(self, prev_layer):
        self.gradients = prev_layer
        if self.activation != None:
            prev_layer = prev_layer * self.derivate()
        # if len(prev_layer[0]) > 1 and self.stride == 1:
        #     return self.backpropC4(prev_layer)
        # else:
        return self.backpropC5(prev_layer)
        # return self.backprop2(prev_layer)

    def updateWeights(self, learn_rate):
        gradient = self.gradients
        self.input = numpy.array(self.inp)
        self.filterSet(gradient, learn_rate)
        self.gradients = []
        self.inp = []