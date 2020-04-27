import ILayer as interface
from scipy import ndimage
from scipy import signal
import numpy
import time
from Adam import AdamOptimizer


class Conv(interface.ILayer):
    def __init__(self, prev_layer_size, num_filters, filter_size, stride = 1, padding = 0):
        numpy.random.seed(10+prev_layer_size*num_filters)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias = numpy.random.randint(1, size=(num_filters, prev_layer_size))
        self.filters = numpy.random.randn(num_filters, prev_layer_size, filter_size, filter_size) / (filter_size * filter_size)
        self.gradients = []
        self.inp = []
        self.adam = AdamOptimizer(self.filters)

    def setFilters(self, filters):
        self.filters = filters

    def setBias(self, bias):
        self.bias = bias

    def conv3(self, input):
        if input.ndim <= 2:
            input = input[numpy.newaxis]
        self.depth = input.shape[0]
        self.width = (input.shape[1] + 2 * self.padding - self.filter_size + self.stride) // self.stride
        self.height = (input.shape[2] + 2 * self.padding - self.filter_size + self.stride) // self.stride

        if self.padding > 0:
            tmp = numpy.zeros([input.shape[0], input.shape[1] + self.padding*2, input.shape[2] + self.padding*2])
            tmp[:, self.padding:tmp.shape[1] - self.padding, self.padding:tmp.shape[2] - self.padding] = input
            self.input = tmp
        else:
            self.input = input

        y = numpy.zeros([self.num_filters, self.width, self.height])

        f1 = self.filters
        self.filters = numpy.flip(self.filters, 3)
        self.filters = numpy.flip(self.filters, 2)

        for k in range(self.num_filters):  # each of the filters
            for d in range(self.depth):  # each slice of the input
                y[k, :, :] += signal.convolve(self.input[d, :, :], self.filters[k, d], mode='valid')
        self.filters = f1
        return y

    def conv2(self, input):
        if input.ndim <= 2:
            input = input[numpy.newaxis]
        self.depth = input.shape[0]
        self.width = (input.shape[1] + 2 * self.padding - self.filter_size + self.stride) // self.stride
        self.height = (input.shape[2] + 2 * self.padding - self.filter_size + self.stride) // self.stride

        if self.padding > 0:
            tmp = numpy.zeros([input.shape[0], input.shape[1] + self.padding*2, input.shape[2] + self.padding*2])
            tmp[:, self.padding:tmp.shape[1] - self.padding, self.padding:tmp.shape[2] - self.padding] = input
            self.input = tmp
        else:
            self.input = input

        y = numpy.zeros([self.num_filters, self.width, self.height])

        f1 = self.filters
        self.filters = numpy.flip(self.filters, 3)
        self.filters = numpy.flip(self.filters, 2)

        f = self.input.shape[1] - self.width
        s = False
        if f % 2 == 1:
            s = True
        for k in range(self.num_filters):  # each of the filters
            for d in range(self.depth):  # each slice of the input
                if s:
                    y[k, :, :] += ndimage.convolve(self.input[d, :, :], self.filters[k, d], mode='constant', cval=0.0)[
                                  (f // 2):-(f // 2) - 1, (f // 2):-(f // 2) - 1]
                else:
                    y[k, :, :] += ndimage.convolve(self.input[d, :, :], self.filters[k, d], mode='constant', cval=0.0)[f//2:-(f//2),f//2:-(f//2)]

        self.filters = f1
        return y

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

    def convolve(self, input):
        if input.ndim <= 2:
            input = input[numpy.newaxis]
        self.depth = input.shape[0]
        self.width = (input.shape[1] + 2 * self.padding - self.filter_size + self.stride) // self.stride
        self.height = (input.shape[2] + 2 * self.padding - self.filter_size + self.stride) // self.stride

        if self.padding > 0:
            tmp = numpy.zeros([input.shape[0], input.shape[1] + self.padding*2, input.shape[2] + self.padding*2])
            tmp[:, self.padding:tmp.shape[1] - self.padding, self.padding:tmp.shape[2] - self.padding] = input
            self.input = tmp
        else:
            self.input = input

        output = numpy.zeros([self.num_filters, self.width, self.height])

        f1 = self.filters
        # self.filters = numpy.flip(self.filters, 2)
        # self.filters = numpy.flip(self.filters, 3)

        for row in range(self.height):
            for column in range(self.width):
                for d in range(self.depth):
                    rows = self.input[d, row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size]
                    for filter in range(self.num_filters):
                        output[filter][row][column] += numpy.sum(rows * self.filters[filter, d]) + self.bias[filter, d]
        self.filters = f1
        return output

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
                for d in range(self.depth):
                    rows = self.input[:, d, row * self.stride: row * self.stride + self.filter_size,
                           column * self.stride:column * self.stride + self.filter_size]
                    for filter in range(self.num_filters):
                        ss = rows * fils[:, filter, d]
                        dd = numpy.sum(ss, axis=(1, 2))
                        output[:, filter, row, column] += dd
        return output

    def backpropC(self, d_L_d_out):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        # Pad d_L_d_out to size of input
        d_out_pad = numpy.zeros([d_L_d_out.shape[0], self.input.shape[1], self.input.shape[2]])
        pad = (d_out_pad.shape[1] - d_L_d_out.shape[1]) // 2
        if (d_out_pad.shape[1] - d_L_d_out.shape[1]) % 2 == 1:
            d_out_pad[:, pad:d_out_pad.shape[1] - pad - 1, pad:d_out_pad.shape[2] - pad - 1] = d_L_d_out
        else:
            d_out_pad[:, pad:d_out_pad.shape[1] - pad, pad:d_out_pad.shape[2] - pad] = d_L_d_out

        d_L_d_input = numpy.zeros(self.input.shape)
        for row in range(self.height):
            for column in range(self.width):
                for filter in range(self.num_filters):
                    rows_i = d_out_pad[filter, row * self.stride: row * self.stride + self.filter_size,
                             column * self.stride:column * self.stride + self.filter_size]
                    for d in range(self.depth):
                        d_L_d_input[d][row][column] += numpy.sum(rows_i * self.filters[filter, d]) + self.bias[filter, d]

        return d_L_d_input

    def backpropC2(self, d_L_d_out):
        d_out_pad = numpy.zeros([d_L_d_out.shape[0], self.input.shape[1], self.input.shape[2]])
        pad = (d_out_pad.shape[1] - d_L_d_out.shape[1]) // 2
        if (d_out_pad.shape[1] - d_L_d_out.shape[1]) % 2 == 1:
            d_out_pad[:, pad:d_out_pad.shape[1] - pad - 1, pad:d_out_pad.shape[2] - pad - 1] = d_L_d_out
        else:
            d_out_pad[:, pad:d_out_pad.shape[1] - pad, pad:d_out_pad.shape[2] - pad] = d_L_d_out

        d_L_d_input = numpy.zeros(self.input.shape)
        for filter in range(self.num_filters):
            for d in range(self.depth):
                    d_L_d_input[d, :, :] += ndimage.convolve(d_out_pad[filter, :, :], self.filters[filter, d])

        return d_L_d_input

    def backpropC3(self, d_L_d_out):
        d_out_pad = numpy.zeros([d_L_d_out.shape[0], self.input.shape[1], self.input.shape[2]])
        pad = (d_out_pad.shape[1] - d_L_d_out.shape[1]) // 2
        if (d_out_pad.shape[1] - d_L_d_out.shape[1]) % 2 == 1:
            d_out_pad[:, pad:d_out_pad.shape[1] - pad - 1, pad:d_out_pad.shape[2] - pad - 1] = d_L_d_out
        else:
            d_out_pad[:, pad:d_out_pad.shape[1] - pad, pad:d_out_pad.shape[2] - pad] = d_L_d_out

        d_L_d_input = numpy.zeros(self.input.shape)
        for filter in range(self.num_filters):
            for d in range(self.depth):
                    d_L_d_input[d, :, :] += signal.convolve(d_out_pad[filter, :, :], self.filters[filter, d], mode='same')

        return d_L_d_input

    def backpropC4(self, d_L_d_out):
        m = d_L_d_out.shape[0]
        d_out_pad = numpy.zeros([d_L_d_out.shape[0], d_L_d_out.shape[1], self.input.shape[1], self.input.shape[2]])
        pad = (d_out_pad.shape[2] - d_L_d_out.shape[2]) // 2
        if (d_out_pad.shape[2] - d_L_d_out.shape[2]) % 2 == 1:
            d_out_pad[:, :, pad:d_out_pad.shape[2] - pad - 1, pad:d_out_pad.shape[3] - pad - 1] = d_L_d_out
        else:
            d_out_pad[:, :, pad:d_out_pad.shape[2] - pad, pad:d_out_pad.shape[3] - pad] = d_L_d_out

        inps = numpy.array(self.inp)
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
        d_L_d_input = numpy.zeros(inps.shape)
        fils = numpy.repeat(self.filters[numpy.newaxis, :, :, :, :], m, axis=0)

        for row in range(self.height):
            for column in range(self.width):
                for filter in range(self.num_filters):
                    rows_i = d_out_pad[:, filter, row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size]
                    for d in range(self.depth):
                        ss = rows_i * fils[:, filter, d]
                        dd = numpy.sum(ss, axis=(1,2))
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

        # self.filters -= learn_rate * d_L_d_filters
        # self.adam.alpha = learn_rate
        self.filters = self.adam.backward_pass(d_L_d_filters)

    def forward1(self, prev_layer):
        if self.stride > 1:
            result = self.convolve(prev_layer)
        else:
            result = self.conv2(prev_layer)
        # self.inp.append(self.input)
        return result

    def forward2(self, prev_layer):
        result = self.convolve2(prev_layer)
        self.inp = self.input
        # result = []
        # for i in range(len(prev_layer)):
        #     result.append(self.conv2(prev_layer[i]))
        #     self.inp.append(self.input)
        # result = numpy.array(result)
        return result

    def backprop1(self, prev_layer):
        if self.stride > 1:
            result = self.backpropC(prev_layer)
        else:
            result = self.backpropC2(prev_layer)
        self.gradients.append(prev_layer)
        return result

    def backprop2(self, prev_layer):
        result = self.backpropC5(prev_layer)
        self.gradients = prev_layer
        return result

    def forward(self, prev_layer):
        if (prev_layer.ndim > 3):
            return self.forward2(prev_layer)
        else:
            return self.forward1(prev_layer)

    def backward(self, prev_layer):
        return self.backprop2(prev_layer)

    def updateWeights(self, learn_rate):
        gradient = self.gradients
        self.input = numpy.array(self.inp)
        self.filterSet(gradient, learn_rate)
        self.gradients = []
        self.inp = []