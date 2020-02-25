import ILayer as interface
import random
import numpy


class Conv(interface.ILayer):
    def __init__(self, num_filters, filter_size, stride = 1, padding = 0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias = numpy.random.randint(2, size=num_filters)
        self.filters = numpy.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)

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
            self.width = (input.shape[0] + 2 * self.padding - self.filter_size + self.stride) // self.stride
            self.height = (input.shape[1] + 2 * self.padding - self.filter_size + self.stride) // self.stride
            self.depth = 1
        else:
            self.depth = input.shape[0]
            self.width = (input.shape[1] + 2 * self.padding - self.filter_size + self.stride) // self.stride
            self.height = (input.shape[2] + 2 * self.padding - self.filter_size + self.stride) // self.stride
        if self.padding > 0:
            tmp = numpy.zeros([input.shape[0] + self.padding*2, input.shape[1] + self.padding*2])
            tmp[self.padding:tmp.shape[0] - self.padding, self.padding:tmp.shape[1] - self.padding] = input
            self.input = tmp
        else:
            self.input = input

        output = numpy.zeros([self.num_filters, self.width + self.padding*2, self.height + self.padding*2])

        for row in range(self.height):
            for column in range(self.width):
                rows = self.input[row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size]
                for filter in range(len(self.filters)):
                    output[filter][row][column] = numpy.sum(rows * self.filters[filter]) + self.bias[filter]
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
                rows = self.input[row * self.stride: row * self.stride + self.filter_size,
                       column * self.stride:column * self.stride + self.filter_size]
                for filter in range(len(self.filters)):
                        d_L_d_filters[filter] += d_L_d_out[filter, row, column] * rows

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # We aren't returning anything here since we use Conv3x3 as
        # the first layer in our CNN. Otherwise, we'd need to return
        # the loss gradient for this layer's inputs, just like every
        # other layer in our CNN.
        return None

    def forward(self, prev_layer):
        return self.convlove(prev_layer)

    def backward(self, prev_layer):
        raise NotImplementedError