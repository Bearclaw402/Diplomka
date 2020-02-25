import ILayer as interface
import numpy

class Pool(interface.ILayer):
    def __init__(self, stride=2, filter_size=2):
        self.stride = stride
        self.filter_size = filter_size

    def maxPooling(self, input):
        width = (len(input[0][0]) - self.filter_size + self.stride) // self.stride
        height = (len(input[0]) - self.filter_size + self.stride) // self.stride
        depth = len(input)
        output = []
        for i in range(depth):
            output.append([[0 for k in range(width)] for j in range(height)])
        for i in range(depth):
            for row in range(height):
                for column in range(width):
                    rows = input[i][row * self.stride:row * self.stride + self.filter_size]
                    partialMatrix = []
                    for j in rows:
                        partialMatrix.append(j[column * self.stride:column * self.stride + self.filter_size])
                    output[i][row][column] = numpy.max(partialMatrix)
        return output

    def forward(self, prev_layer):
        return self.maxPooling(prev_layer)

    def backward(self, prev_layer):
        raise NotImplementedError