import ILayer as interface
import numpy


class Pool(interface.ILayer):
    def __init__(self, stride=2, filter_size=2, pool_type='max'):
        self.stride = stride
        self.filter_size = filter_size
        self.pool_type = pool_type
        self.mask = []

    def maxPooling(self, input):
        self.last_input = input
        output = numpy.zeros([self.depth, self.height, self.width])
        self.output_indexes = numpy.zeros(output.shape)
        for i in range(self.depth):
            for row in range(self.height):
                for column in range(self.width):
                    partial_matrix = input[i][row * self.stride: row * self.stride + self.filter_size,
                                    column * self.stride:column * self.stride + self.filter_size]
                    output[i][row][column] = numpy.max(partial_matrix)
                    self.output_indexes[i][row][column] = numpy.argmax(partial_matrix)
        return output

    def avgPooling(self, input):
        self.last_input = input
        output = numpy.zeros([self.depth, self.height, self.width])
        for i in range(self.depth):
            for row in range(self.height):
                for column in range(self.width):
                    partial_matrix = input[i][row * self.stride: row * self.stride + self.filter_size,
                                    column * self.stride:column * self.stride + self.filter_size]
                    output[i][row][column] = numpy.average(partial_matrix)
        return output

    def medPooling(self, input):
        self.last_input = input
        output = numpy.zeros([self.depth, self.height, self.width])
        for row in range(self.height):
            for column in range(self.width):
                partial_matrix = input[:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size]
                output[:,row,column] = numpy.median(partial_matrix, axis=(1,2))
                print()
        return output

    def weightAvgPooling(self, input):
        self.last_input = input
        output = numpy.zeros([input.shape[0],self.depth, self.height, self.width])
        self.output_indexes = numpy.zeros(output.shape)
        weights = numpy.zeros(input.shape)
        for row in range(self.height):
            for column in range(self.width):
                partial_matrix = input[:,:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size]
                output[:,:,row,column] = numpy.mean(partial_matrix)
                partial_matrix+=1
                mmaxx = numpy.max(partial_matrix, axis=(2,3))[:,:,numpy.newaxis, numpy.newaxis]
                weights[:,:,row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size] = partial_matrix / numpy.repeat(numpy.repeat(mmaxx,2,axis=2),2,axis=3)
        self.mask = weights
        return output

    def maxPooling2(self, input):
        self.last_input = input
        output = numpy.zeros([input.shape[0],self.depth, self.height, self.width])
        self.output_indexes = numpy.zeros(output.shape)
        weights = numpy.zeros(input.shape)
        for row in range(self.height):
            for column in range(self.width):
                partial_matrix = input[:,:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size]
                max1 = numpy.max(partial_matrix, axis=(2,3))
                mmaxx = max1[:,:,numpy.newaxis,numpy.newaxis]
                mmaxx = numpy.repeat(numpy.repeat(mmaxx,2,axis=2),2,axis=3)
                indexes1 = partial_matrix >= mmaxx
                partial_matrix[indexes1] = -2.0
                max2 = numpy.max(partial_matrix, axis=(2,3))
                mmaxx = max2[:,:,numpy.newaxis,numpy.newaxis]
                mmaxx = numpy.repeat(numpy.repeat(mmaxx,2,axis=2),2,axis=3)
                indexes2 = partial_matrix >= mmaxx
                output[:,:,row,column] = (max1 + max2) / 2.0
                weights[:,:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size] = (indexes1 + indexes2)
        self.mask = weights
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
        return 0.25*(numpy.repeat(numpy.repeat(d_L_d_out,self.filter_size,axis=1),self.filter_size,axis=2))

    def poolFW1(self, x):
        self.last_input = x
        out = numpy.zeros([self.depth, self.height, self.width])
        if x.shape[1] % 2 == 1:
            x = x[:, : x.shape[1] - 1, : x.shape[2]-1]
        x_patches = x.reshape(x.shape[0], x.shape[1] // self.stride, self.filter_size, x.shape[2] // self.stride, self.filter_size)
        if self.pool_type == "max":
            out = x_patches.max(axis=2).max(axis=3)
            self.mask.append(numpy.isclose(x, numpy.repeat(numpy.repeat(out, self.filter_size, axis=1), self.filter_size, axis=2)).astype(int))
        elif self.pool_type == "avg":
            out = x_patches.mean(axis=2).mean(axis=3)
            self.mask.append(numpy.ones_like(x) * 0.25)
        elif self.pool_type == "med":
            out = numpy.median(x_patches,axis=(2,4))
            self.mask.append(numpy.ones_like(x) * 0.25)
        elif self.pool_type == "wavg":
            self.weightAvgPooling(x)
        elif self.pool_type == "max2":
            self.maxPooling2(x)
        return out

    def poolBW1(self, d_L_d_out):
        if self.pool_type == 'wavg' or self.pool_type == 'max2':
            return self.mask * (numpy.repeat(numpy.repeat(d_L_d_out, self.filter_size, axis=1), self.filter_size, axis=2))
        else:
            return self.mask.pop() * (numpy.repeat(numpy.repeat(d_L_d_out, self.filter_size, axis=1), self.filter_size, axis=2))

    def forward1(self, prev_layer):
        self.width = (prev_layer.shape[1] - self.filter_size + self.stride) // self.stride
        self.height = (prev_layer.shape[2] - self.filter_size + self.stride) // self.stride
        self.depth = prev_layer.shape[0]
        result = numpy.zeros([self.depth, self.height, self.width])
        if self.stride != self.filter_size:
            if self.pool_type == 'max':
                result = self.maxPooling(prev_layer)
            elif self.pool_type == 'avg':
                result = self.avgPooling(prev_layer)
        else:
            result = self.poolFW1(prev_layer)
        return result

    def forward2(self, prev_layer):
        self.width = (prev_layer.shape[2] - self.filter_size + self.stride) // self.stride
        self.height = (prev_layer.shape[3] - self.filter_size + self.stride) // self.stride
        self.depth = prev_layer.shape[1]
        result = numpy.zeros([len(prev_layer), self.depth, self.height, self.width])
        if (self.pool_type == 'wavg'):
            return self.weightAvgPooling(prev_layer)
        elif self.pool_type == 'max2':
            return self.maxPooling2(prev_layer)
        for i in range(len(prev_layer)):
            if self.stride != self.filter_size:
                if self.pool_type == 'max':
                    result[i] = self.maxPooling(prev_layer[i])
                elif self.pool_type == 'avg':
                    result[i] = self.avgPooling(prev_layer[i])
            else:
                    result[i] = self.poolFW1(prev_layer[i])
        return result

    def backprop1(self, d_L_d_out):
        result = numpy.zeros(self.last_input.shape)
        if self.stride != self.filter_size:
            if self.pool_type == 'max':
                result = self.maxBackprop(d_L_d_out)
            elif self.pool_type == 'avg':
                result = self.avgBackprop(d_L_d_out)
        else:
            if self.last_input.shape[1] % 2 == 1:
                result[:, : -1, : -1] = self.poolBW1(d_L_d_out)
            else:
                result = self.poolBW1(d_L_d_out)
        return result

    def backprop2(self, d_L_d_out):
        result = []
        mmm = self.mask
        for i in range(d_L_d_out.shape[0]):
            if self.pool_type == 'wavg' or self.pool_type == 'max2':
                self.mask = mmm[i]
            tmp = numpy.zeros(self.last_input.shape)
            if self.stride != self.filter_size:
                if self.pool_type == 'max':
                    tmp = self.maxBackprop(d_L_d_out[i])
                elif self.pool_type == 'avg':
                    tmp = self.avgBackprop(d_L_d_out[i])
            else:
                if self.last_input.shape[1] % 2 == 1:
                    tmp[:, : -1, : -1] = self.poolBW1(d_L_d_out[i])
                else:
                    tmp = self.poolBW1(d_L_d_out[i])
            result.append(tmp)
        result = numpy.array(result)
        return result

    def forward(self, prev_layer):
        if (prev_layer.ndim > 3):
            return self.forward2(prev_layer)
        else:
            return self.forward1(prev_layer)

    def backward(self, d_L_d_out):
        return self.backprop2(d_L_d_out)

    def updateWeights(self, leran_rate):
        pass