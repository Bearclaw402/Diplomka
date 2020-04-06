import ILayer as interface
from scipy import ndimage
import numpy


class Conv(interface.ILayer):
    def __init__(self, prev_layer_size, num_filters, filter_size, stride = 1, padding = 0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias = numpy.random.randint(1, size=(num_filters, prev_layer_size))
        self.filters = numpy.random.randn(num_filters, prev_layer_size, filter_size, filter_size) / (filter_size * filter_size)

    def setFilters(self, filters):
        self.filters = filters

    def setBias(self, bias):
        self.bias = bias

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
    #     #
        y = numpy.zeros([self.num_filters, self.width, self.height])
    #     #
        f1 = self.filters
        self.filters = numpy.flip(self.filters, 3)
        self.filters = numpy.flip(self.filters, 2)

        f = self.filter_size
        s = False
        if f // 2 - (f - f // 2) == 0:
            s = True
        for k in range(self.num_filters):  # each of the filters
            for d in range(self.depth):  # each slice of the input
                if s:
                    dd =  numpy.flip(self.filters, 2)
                    dd =  numpy.flip(dd, 3)
                    y2 = numpy.sum(self.input[d, :, :] * dd[k, d])
                    y[k, :, :] += y2
                else:
                    y[k, :, :] += ndimage.convolve(self.input[d, :, :], self.filters[k, d], mode='constant', cval=0.0)[f//2:-(f//2),f//2:-(f//2)]

        self.filters = f1
        return y

    def convlove(self, input):
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

        # self.filters = numpy.flip(self.filters, 2)
        # self.filters = numpy.flip(self.filters, 3)

        for row in range(self.height):
            for column in range(self.width):
                for d in range(self.depth):
                    rows = self.input[d, row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size]
                    for filter in range(self.num_filters):
                        output[filter][row][column] += numpy.sum(rows * self.filters[filter, d]) + self.bias[filter, d]
        return output

    def backpropC(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        # Pad d_L_d_out to size of input
        d_out_pad = numpy.zeros([d_L_d_out.shape[0], self.input.shape[1], self.input.shape[2]])
        pad = (d_out_pad.shape[1] - d_L_d_out.shape[1]) // 2
        d_out_pad[:, pad:d_out_pad.shape[1] - pad, pad:d_out_pad.shape[2] - pad] = d_L_d_out

        d_L_d_input = numpy.zeros(self.input.shape)
        for filter in range(self.num_filters):
            for d in range(self.depth):
                    d_L_d_input[d, :, :] += ndimage.convolve(d_out_pad[filter, :, :], self.filters[filter, d])

        return d_L_d_input

    def filterSet(self, d_L_d_out, learn_rate):
        x = numpy.moveaxis(self.input, 0, -1)
        x_padded_bcast = numpy.expand_dims(x, axis=-1)  # shape = (w, h, d, 1)
        dz = numpy.moveaxis(d_L_d_out, 0, -1)
        dZ_bcast = numpy.expand_dims(dz, axis=-2)  # shape = (i, j, 1, k)
        d_L_d_filters = numpy.zeros(self.filters.shape)
        for a in range(self.filter_size):
            for b in range(self.filter_size):
                asd = numpy.sum(
                    dZ_bcast * x_padded_bcast[a:a + self.width,
                               b:b + self.height, :, :], axis=(0, 1))
                d_L_d_filters[:, :, a, b] = asd.swapaxes(0,1)

        self.filters -= learn_rate * d_L_d_filters

    def forward(self, prev_layer):
        result = self.conv2(prev_layer)
        return result

    def backward(self, prev_layer, leran_rate):
        result = self.backpropC(prev_layer, leran_rate)
        self.filterSet(prev_layer, leran_rate)
        return result