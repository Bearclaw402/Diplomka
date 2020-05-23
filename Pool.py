import ILayer as interface
import numpy


class Pool(interface.ILayer):
    def __init__(self, stride=2, filter_size=2, pool_type='max'):
        self.stride = stride
        self.filter_size = filter_size
        self.pool_type = pool_type
        self.mask = []

    def __max_pooling__(self, input):
        self.last_input = input
        output = numpy.zeros([input.shape[0],self.depth, self.height, self.width])
        weights = numpy.zeros((input.shape[0],input.shape[1],input.shape[2]-1,input.shape[3]-1))
        for row in range(self.height):
            for column in range(self.width):
                partial_matrix = input[:,:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size]
                max1 = numpy.max(partial_matrix, axis=(2,3))
                mmaxx = max1[:,:,numpy.newaxis,numpy.newaxis]
                mmaxx = numpy.repeat(numpy.repeat(mmaxx,self.filter_size,axis=2),self.filter_size,axis=3)
                indexes1 = partial_matrix >= mmaxx
                output[:,:,row,column] = max1
                weights[:,:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size] = indexes1
        self.mask = weights
        return output

    def __avg_pooling__(self, input):
        self.last_input = input
        output = numpy.zeros([input.shape[0],self.depth, self.height, self.width])
        weights = numpy.ones((input.shape[0],input.shape[1],input.shape[2]-1,input.shape[3]-1))
        for row in range(self.height):
            for column in range(self.width):
                partial_matrix = input[:,:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size]
                output[:,:,row,column] = numpy.mean(partial_matrix, axis=(2,3))
        self.mask = weights * 0.25
        return output

    def __med_pooling__(self, input):
        self.last_input = input
        output = numpy.zeros([input.shape[0], self.depth, self.height, self.width])
        weights = numpy.ones((input.shape[0], input.shape[1], input.shape[2]-1, input.shape[3]-1))
        for row in range(self.height):
            for column in range(self.width):
                partial_matrix = input[:,:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size]
                output[:,:,row,column] = numpy.median(partial_matrix, axis=(2,3))
        self.mask = weights * 0.25
        return output

    def __wavg_pooling__(self, input):
        self.last_input = input
        output = numpy.zeros([input.shape[0],self.depth, self.height, self.width])
        self.output_indexes = numpy.zeros(output.shape)
        weights = numpy.zeros((input.shape[0],input.shape[1],input.shape[2]-1,input.shape[3]-1))
        for row in range(self.height):
            for column in range(self.width):
                partial_matrix = input[:,:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size]
                output[:,:,row,column] = numpy.mean(partial_matrix, axis=(2,3))
                partial_matrix+=1
                mmaxx = numpy.max(partial_matrix, axis=(2,3))[:,:,numpy.newaxis, numpy.newaxis]
                weights[:,:,row * self.stride: row * self.stride + self.filter_size, column * self.stride:column * self.stride + self.filter_size] = partial_matrix / numpy.repeat(numpy.repeat(mmaxx,self.filter_size,axis=2),self.filter_size,axis=3)
        self.mask = weights
        return output

    def __max2_pooling__(self, input):
        self.last_input = input
        output = numpy.zeros([input.shape[0],self.depth, self.height, self.width])
        self.output_indexes = numpy.zeros(output.shape)
        weights = numpy.zeros((input.shape[0],input.shape[1],input.shape[2]-1,input.shape[3]-1))
        for row in range(self.height):
            for column in range(self.width):
                partial_matrix = input[:,:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size]
                max1 = numpy.max(partial_matrix, axis=(2,3))
                mmaxx = max1[:,:,numpy.newaxis,numpy.newaxis]
                mmaxx = numpy.repeat(numpy.repeat(mmaxx,self.filter_size,axis=2),self.filter_size,axis=3)
                indexes1 = partial_matrix >= mmaxx
                partial_matrix[indexes1] = -2.0
                max2 = numpy.max(partial_matrix, axis=(2,3))
                mmaxx = max2[:,:,numpy.newaxis,numpy.newaxis]
                mmaxx = numpy.repeat(numpy.repeat(mmaxx,self.filter_size,axis=2),self.filter_size,axis=3)
                indexes2 = partial_matrix >= mmaxx
                output[:,:,row,column] = (max1 + max2) / 2.0
                weights[:,:,row * self.stride: row * self.stride + self.filter_size,
                                     column * self.stride:column * self.stride + self.filter_size] = (indexes1 + indexes2)
        self.mask = weights
        return output

    def poolBW(self, d_L_d_out):
        result =  self.mask * (numpy.repeat(numpy.repeat(d_L_d_out, self.filter_size, axis=2), self.filter_size, axis=3))
        return numpy.array(result)

    def poolFW(self, prev_layer):
        self.width = (prev_layer.shape[2] - self.filter_size + self.stride) // self.stride
        self.height = (prev_layer.shape[3] - self.filter_size + self.stride) // self.stride
        self.depth = prev_layer.shape[1]
        return self.__getattribute__("__" + self.pool_type + "_pooling__")(prev_layer)

    def forward(self, prev_layer):
        return self.poolFW(prev_layer)

    def backward(self, d_L_d_out):
        return self.poolBW(d_L_d_out)

    def updateWeights(self, learn_rate, optimizer):
        pass