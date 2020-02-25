import ILayer as interface
import random
import numpy


class Conv(interface.ILayer):
    def __init__(self, num_filters, filter_size, stride = 1, padding = 0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        #self.filters = []
        self.bias = []
        self.filters = numpy.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
        for i in range(num_filters):
            #self.filters.append([[random.random() for k in range(filter_size)] for j in range(filter_size)])

            self.bias.append(0 if random.random() < 0.5 else 1)

    def setFilters(self, filters):
        self.filters = filters

    def setBias(self, bias):
        self.bias = bias

    def zeros(self, width, height, output):
        for i in range(self.num_filters):
            output.append([[0 for k in range(width + self.padding*2)] for j in range(height + self.padding*2)])
        return output

    def convlove(self, input):
        #width = (len(input[0][0]) + 2 * self.padding - self.filter_size + self.stride) // self.stride
        #height = (len(input[0]) + 2 * self.padding - self.filter_size + self.stride) // self.stride
        #depth = len(input)
        width = (input.shape[0] + 2 * self.padding - self.filter_size + self.stride) // self.stride
        height = (input.shape[1] + 2 * self.padding - self.filter_size + self.stride) // self.stride
        depth = 1 if input.ndim <= 2 else input.shape[2]
        output = []
        #output = numpy.zeros(input.shape)
        output = self.zeros(width, height, output)
        # for i in range(self.num_filters):
        #     output.append([[0 for k in range(width)] for j in range(height)])
        for filter in range(len(self.filters)):
            for row in range(height):
                for column in range(width):
                    result = 0
                    for i in range(depth):
                        # rows = []
                        # firstZeros = (self.padding - (row * self.stride)) if (
                        #             self.padding - (row * self.stride)) < self.filter_size else self.filter_size
                        # firstZeros = 0 if firstZeros < 0 else firstZeros
                        # #lastZeros = (self.padding - (len(input[0]) + 2 * self.padding - (row * self.stride + self.filter_size))) if (
                        # #           self.padding - (len(input[0]) + 2 * self.padding - (row * self.stride + self.filter_size))) < self.filter_size else self.filter_size
                        # lastZeros = (self.padding - (input.shape[1] + 2 * self.padding - (row * self.stride + self.filter_size))) if (
                        #            self.padding - (input.shape[1] + 2 * self.padding - (row * self.stride + self.filter_size))) < self.filter_size else self.filter_size
                        # lastZeros = 0 if lastZeros < 0 else lastZeros
                        # leftZeros = (self.padding - (column * self.stride)) if (
                        #              self.padding - (column * self.stride)) < self.filter_size else self.filter_size
                        # leftZeros = 0 if leftZeros < 0 else leftZeros
                        # #rightZeros = (self.padding - (len(input[0][0]) + 2 * self.padding - (column * self.stride + self.filter_size))) if (
                        # #              self.padding - (len(input[0][0]) + 2 * self.padding - (column * self.stride + self.filter_size))) < self.filter_size else self.filter_size
                        # rightZeros = (self.padding - (input.shape[0] + 2 * self.padding - (column * self.stride + self.filter_size))) if (
                        #               self.padding - (input.shape[0] + 2 * self.padding - (column * self.stride + self.filter_size))) < self.filter_size else self.filter_size
                        # rightZeros = 0 if rightZeros < 0 else rightZeros
                        # beginRow = (row * self.stride + firstZeros - self.padding) if (
                        #             row * self.stride + firstZeros - self.padding) > 0 else 0
                        # endRow = ((row * self.stride + self.filter_size) - lastZeros - self.padding) if ((
                        #            row * self.stride + self.filter_size) - lastZeros - self.padding) > 0 else 0
                        # beginCol = (column * self.stride + leftZeros - self.padding) if (
                        #             column * self.stride + leftZeros - self.padding) > 0 else 0
                        # endCol = ((column * self.stride + self.filter_size) - rightZeros - self.padding) if ((
                        #            column * self.stride + self.filter_size) - rightZeros - self.padding) > 0 else 0
                        # for j in range(firstZeros):
                        #     rows.append([0 for k in range(self.filter_size)])
                        # for j in range(endRow - beginRow):
                        #     tmpRow = []
                        #     for k in range(leftZeros):
                        #         tmpRow.append(0)
                        #     for k in range(endCol - beginCol):
                        #         #tmpRow.append(input[i][beginRow + j][beginCol + k])
                        #         tmpRow.append(input[beginRow + j][beginCol + k])
                        #     for k in range(rightZeros):
                        #         tmpRow.append(0)
                        #     rows.append(tmpRow)
                        # for j in range(lastZeros):
                        #     rows.append([0 for k in range(self.filter_size)])
                        rows = input[row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size]
                        #result += numpy.sum(numpy.array(rows) * self.filters[filter][i])
                        result += numpy.sum(numpy.array(rows) * self.filters[filter])
                    #print(filter, row, column)
                    output[filter][row][column] = result + self.bias[filter]
        return output

    def forward(self, prev_layer):
        return self.convlove(prev_layer)

    def backward(self, prev_layer):
        raise NotImplementedError