import ILayer as interface
import numpy


class Pool(interface.ILayer):
    def __init__(self, stride=2, filter_size=2, pool_type='max'):
        self.stride = stride
        self.filter_size = filter_size
        self.pool_type = pool_type

    def maxPooling(self, input):
        self.last_input = input
        width = (input.shape[1] - self.filter_size + self.stride) // self.stride
        height = (input.shape[2] - self.filter_size + self.stride) // self.stride
        depth = input.shape[0]
        output = numpy.zeros([depth, height, width])
        self.output_indexes = numpy.zeros(output.shape)
        for i in range(depth):
            for row in range(height):
                for column in range(width):
                    partial_matrix = input[i][row * self.stride: row * self.stride + self.filter_size,
                                    column * self.stride:column * self.stride + self.filter_size]
                    output[i][row][column] = numpy.max(partial_matrix)
                    self.output_indexes[i][row][column] = numpy.argmax(partial_matrix)
        return output

    def avgPooling(self, input):
        self.last_input = input
        width = (input.shape[1] - self.filter_size + self.stride) // self.stride
        height = (input.shape[2] - self.filter_size + self.stride) // self.stride
        depth = input.shape[0]
        output = numpy.zeros([depth, height, width])
        for i in range(depth):
            for row in range(height):
                for column in range(width):
                    partial_matrix = input[i][row * self.stride: row * self.stride + self.filter_size,
                                    column * self.stride:column * self.stride + self.filter_size]
                    output[i][row][column] = numpy.average(partial_matrix)
        return output

    def maxBackprop(self, d_L_d_out):
        '''
            Performs a backward pass of the maxpool layer.
            Returns the loss gradient for this layer's inputs.
            - d_L_d_out is the loss gradient for this layer's outputs.
        '''
        d_L_d_input = numpy.zeros(self.last_input.shape)

        d, w, h = d_L_d_out.shape

        for i in range(d):
            for row in range(h):
                for column in range(w):
                    row2 = int(self.output_indexes[i][row][column] // self.filter_size)
                    column2 = int(self.output_indexes[i][row][column] % self.filter_size)

                    d_L_d_input[i][row * self.stride + row2][column * self.stride + column2] = d_L_d_out[i][row][column]

        return d_L_d_input

    def avgBackprop(self, d_L_d_out):
        return 0.25*(numpy.repeat(numpy.repeat(d_L_d_out,2,axis=1),2,axis=2))

    def forward(self, prev_layer):
        if self.pool_type == 'max':
            return self.maxPooling(prev_layer)
        elif self.pool_type == 'avg':
            return self.avgPooling(prev_layer)

    def backward(self, d_L_d_out, leran_rate):
        if self.pool_type == 'max':
            return self.maxBackprop(d_L_d_out)
        elif self.pool_type == 'avg':
            return self.avgBackprop(d_L_d_out)