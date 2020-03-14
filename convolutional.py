import ILayer as interface
import random
import numpy


class Conv(interface.ILayer):
    def __init__(self, prev_layer_size, num_filters, filter_size, stride = 1, padding = 0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        # self.bias = numpy.random.randint(2, size=(num_filters, prev_layer_size))
        self.bias = numpy.random.randint(1, size=(num_filters, prev_layer_size))
        self.filters = numpy.random.randn(num_filters, prev_layer_size, filter_size, filter_size) / (filter_size * filter_size)

    def setFilters(self, filters):
        self.filters = filters

    def setBias(self, bias):
        self.bias = bias

    def zeros(self, width, height, output):
        for i in range(self.num_filters):
            output.append([[0 for k in range(width + self.padding*2)] for j in range(height + self.padding*2)])
        return output

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

        self.filters = numpy.flip(self.filters, 0)
        self.filters = numpy.flip(self.filters, 1)

        for row in range(self.height):
            for column in range(self.width):
                for d in range(self.depth):
                    rows = self.input[d, row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size]
                    for filter in range(self.num_filters):
                        output[filter][row][column] += numpy.sum(rows * self.filters[filter, d]) + self.bias[filter, d]
        return output

    def backprop(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        d_L_d_filters = numpy.zeros(self.filters.shape)

        for row in range(self.height):
            for column in range(self.width):
                for d in range(self.depth):
                    rows = self.input[d, row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size]
                    for filter in range(self.num_filters):
                            d_L_d_filters[filter, d] += d_L_d_out[filter, row, column] * rows

        # Pad d_L_d_out to size of input
        d_out_pad = numpy.zeros([d_L_d_out.shape[0], self.input.shape[1] + self.filter_size - self.stride, self.input.shape[2] + self.filter_size - self.stride])
        pad = (d_out_pad.shape[1] - d_L_d_out.shape[1]) // 2
        ddd = d_out_pad[:, pad:d_out_pad.shape[1] - pad, pad:d_out_pad.shape[2] - pad]
        d_out_pad[:, pad:d_out_pad.shape[1] - pad, pad:d_out_pad.shape[2] - pad] = d_L_d_out

        d_L_d_input = numpy.zeros(self.input.shape)

        for row in range(self.height):
            for column in range(self.width):
                for d in range(self.depth):
                    for filter in range(self.num_filters):
                        rows = d_out_pad[filter, row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size]
                        d_L_d_input[d][row][column] += numpy.sum(rows * self.filters[filter, d]) + self.bias[filter, d]

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        return d_L_d_input

    def forward(self, prev_layer):
        return self.convlove(prev_layer)

    def backward(self, prev_layer):
        raise NotImplementedError