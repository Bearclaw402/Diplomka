import ILayer as interface
import numpy
from Initializer import Initializer
from Optimizer import AdamOptimizer


class Conv(interface.ILayer):
    def __init__(self, prev_layer_size, num_filters, filter_size, stride=1, padding=0, initializer='xavier_uniform',
                 activation=None):
        numpy.random.seed(10 + prev_layer_size * num_filters)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias = numpy.random.randint(1, size=(num_filters, prev_layer_size))
        self.activation = activation
        n_in = self.filter_size * self.filter_size * prev_layer_size
        n_out = self.filter_size * self.filter_size * self.num_filters
        shape = [self.num_filters, prev_layer_size, self.filter_size, self.filter_size]
        init = Initializer(n_in, n_out, shape, initializer)
        self.filters = init.initializeWeights()
        self.gradients = []
        self.inp = []
        self.output = []
        self.adam = AdamOptimizer(self.filters)

    def activate(self, output):
        self.output = self.__getattribute__("__activate_" + self.activation + "__")(output)
        return self.output

    @staticmethod
    def __activate_relu__(output):
        output[output < 0] = 0
        return output

    @staticmethod
    def __activate_tanh__(output):
        return (numpy.exp(output) - numpy.exp(-output)) / (numpy.exp(output) + numpy.exp(-output))

    @staticmethod
    def __activate_sigmoid__(output):
        return 1.0 / (1 + numpy.exp(-output))

    @staticmethod
    def __activate_cbrt__(potential):
        return numpy.cbrt(potential)

    @staticmethod
    def __activate_comb__(potential):
        result = potential
        index1 = potential >= 0
        index2 = potential < 0
        result[index1] = (numpy.exp(potential[index1]) - numpy.exp(-potential[index1])) / (
                    numpy.exp(potential[index1]) + numpy.exp(-potential[index1]))
        result[index2] = 0.01 * potential[index2]
        return result

    @staticmethod
    def __activate_comb2__(potential):
        result = potential
        index1 = potential >= 0
        index2 = potential < 0
        result[index1] = numpy.sqrt(potential[index1])
        result[index2] = numpy.cbrt(potential[index2])
        return result

    @staticmethod
    def __activate_log__(potential):
        return ((numpy.exp(potential / 2.0) - numpy.exp(-potential / 2.0)) / (
                    numpy.exp(potential / 2.0) + numpy.exp(-potential / 2.0)) / 2.0) + 0.5

    def derivate(self):
        return self.__getattribute__("__derivative_" + self.activation + "__")(self.output)

    @staticmethod
    def __derivative_relu__(activation):
        activation[activation > 0] = 1
        activation[activation <= 0] = 0
        return activation

    @staticmethod
    def __derivative_tanh__(activation):
        return 1 - activation ** 2

    @staticmethod
    def __derivative_sigmoid__(activation):
        return activation * (1.0 - activation)

    @staticmethod
    def __derivative_cbrt__(activation):
        return 1.0 / (3 * numpy.cbrt(numpy.power(activation, 2)))

    @staticmethod
    def __derivative_comb__(activation):
        result = activation
        index1 = activation >= 0
        index2 = activation < 0
        result[index1] = 1 - activation[index1] ** 2
        result[index2] = 0.01
        return result

    @staticmethod
    def __derivative_comb2__(activation):
        result = activation
        index1 = activation > 0
        index2 = activation < 0
        result[index1] = 1.0 / (2 * numpy.sqrt(activation[index1]))
        result[index2] = 1.0 / (3 * numpy.cbrt(numpy.power(activation[index2], 2)))
        return result

    @staticmethod
    def __derivative_log__(activation):
        return (activation * (1.0 - activation))

    def convolve2(self, input):
        m = input.shape[0]
        if input.ndim <= 3:
            input = input[:, numpy.newaxis, :, :]
        self.depth = input.shape[1]
        self.width = (input.shape[2] + 2 * self.padding - self.filter_size + self.stride) // self.stride
        self.height = (input.shape[3] + 2 * self.padding - self.filter_size + self.stride) // self.stride

        if self.padding > 0:
            tmp = numpy.zeros(
                [input.shape[0], input.shape[1], input.shape[2] + self.padding * 2, input.shape[3] + self.padding * 2])
            tmp[:, :, self.padding:tmp.shape[2] - self.padding, self.padding:tmp.shape[3] - self.padding] = input
            self.input = tmp
        else:
            self.input = input

        output = numpy.zeros([m, self.num_filters, self.width, self.height])

        fils = numpy.repeat(self.filters[numpy.newaxis, :, :, :, :], m, axis=0)

        for row in range(self.height):
            for column in range(self.width):
                rows = self.input[:, :, row * self.stride: row * self.stride + self.filter_size,
                       column * self.stride:column * self.stride + self.filter_size]
                for filter in range(self.num_filters):
                    ss = rows * fils[:, filter, :]
                    dd = numpy.sum(ss, axis=(1, 2, 3))
                    output[:, filter, row, column] = dd
        return output

    def backprop(self, d_L_d_out):
        m = d_L_d_out.shape[0]
        d_out_pad = numpy.zeros([d_L_d_out.shape[0], d_L_d_out.shape[1], self.input.shape[2], self.input.shape[3]])
        pad = (d_out_pad.shape[2] - d_L_d_out.shape[2]) // 2
        if (d_out_pad.shape[2] - d_L_d_out.shape[2]) % 2 == 1:
            d_out_pad[:, :, pad:d_out_pad.shape[2] - pad - 1, pad:d_out_pad.shape[3] - pad - 1] = d_L_d_out
        else:
            d_out_pad[:, :, pad:d_out_pad.shape[2] - pad, pad:d_out_pad.shape[3] - pad] = d_L_d_out

        inps = numpy.array(self.inp)
        d_L_d_input = numpy.zeros(inps.shape)
        fils = numpy.repeat(self.filters[numpy.newaxis, :, :, :, :], m, axis=0)

        for row in range(self.height):
            for column in range(self.width):
                rows_i = d_out_pad[:, :, row * self.stride: row * self.stride + self.filter_size,
                         column * self.stride:column * self.stride + self.filter_size]
                for d in range(self.depth):
                    ss = rows_i * fils[:, :, d]
                    dd = numpy.sum(ss, axis=(1, 2, 3))
                    d_L_d_input[:, d, row, column] += dd

        d_L_d_input = d_L_d_input / m
        return d_L_d_input

    def filterSet(self, d_L_d_out):
        x = numpy.moveaxis(self.input, 1, -1)
        x_padded_bcast = numpy.expand_dims(x, axis=-1)
        dz = numpy.moveaxis(d_L_d_out, 1, -1)
        dZ_bcast = numpy.expand_dims(dz, axis=-2)
        d_L_d_filters = numpy.zeros(self.filters.shape)
        for a in range(self.filter_size):
            for b in range(self.filter_size):
                asd = 1 / len(self.gradients) * numpy.sum(
                    dZ_bcast * x_padded_bcast[:, a:a + self.width,
                               b:b + self.height, :, :], axis=(0, 1, 2))
                d_L_d_filters[:, :, a, b] = asd.swapaxes(0, 1)
        self.filters = self.adam.backward_pass(d_L_d_filters)

    def forward(self, prev_layer):
        self.inp = prev_layer
        result = self.convolve2(prev_layer)
        if self.activation is not None:
            result = self.activate(result)
        return result

    def backward(self, prev_layer):
        self.gradients = prev_layer
        if self.activation is not None:
            prev_layer = prev_layer * self.derivate()
        return self.backprop(prev_layer)

    def updateWeights(self, learn_rate):
        gradient = self.gradients
        self.input = numpy.array(self.inp)
        self.filterSet(gradient)
        self.gradients = []
        self.inp = []
