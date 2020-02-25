import ILayer as interface
import numpy


class Pool(interface.ILayer):
    def __init__(self, stride=2, filter_size=2):
        self.stride = stride
        self.filter_size = filter_size

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
                    partialMatrix = input[i][row * self.stride: row * self.stride + self.filter_size,
                                    column * self.stride:column * self.stride + self.filter_size]
                    output[i][row][column] = numpy.max(partialMatrix)
                    self.output_indexes[i][row][column] = numpy.argmax(partialMatrix)
        self.last_output = output
        return output

    def forward(self, prev_layer):
        return self.maxPooling(prev_layer)

    def backward(self, d_L_d_out):
        '''
            Performs a backward pass of the maxpool layer.
            Returns the loss gradient for this layer's inputs.
            - d_L_d_out is the loss gradient for this layer's outputs.
            '''
        d_L_d_input = numpy.zeros(self.last_input.shape)

        d, w, h = self.last_output.shape

        for i in range(d):
            for row in range(h):
                for column in range(w):
                    row2 = int(self.output_indexes[i][row][column] // self.filter_size)
                    column2 = int(self.output_indexes[i][row][column] % self.filter_size)

                    d_L_d_input[i][row * self.stride + row2][column * self.stride + column2] = self.last_output[i][row][column]

        # TEST
        # width = (self.last_input.shape[1] - self.filter_size + self.stride) // self.stride
        # height = (self.last_input.shape[2] - self.filter_size + self.stride) // self.stride
        # depth = self.last_input.shape[0]
        # output = numpy.zeros([depth, height, width])
        # self.output_indexes = numpy.zeros(output.shape)
        # for i in range(depth):
        #     for row in range(height):
        #         for column in range(width):
        #             partialMatrix = self.last_input[i][row * self.stride: row * self.stride + self.filter_size,
        #                             column * self.stride:column * self.stride + self.filter_size]
        #
        #             partialMatrix2 = d_L_d_input[i][row * self.stride: row * self.stride + self.filter_size,
        #                             column * self.stride:column * self.stride + self.filter_size]
        #             if (numpy.max(partialMatrix) < 0):
        #                 if (numpy.max(partialMatrix) == numpy.min(partialMatrix2)):
        #                     if (numpy.argmax(partialMatrix) == numpy.argmin(partialMatrix2)):
        #                         continue
        #                     else:
        #                         print("do riti 2")
        #                 else:
        #                     print("do riti")
        #             else:
        #                 if (numpy.max(partialMatrix) == numpy.max(partialMatrix2)):
        #                     if (numpy.argmax(partialMatrix) == numpy.argmax(partialMatrix2)):
        #                         continue
        #                     else:
        #                         print("do riti 2")
        #                 else:
        #                     print("do riti")

        return d_L_d_input