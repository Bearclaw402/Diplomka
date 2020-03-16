from abc import abstractmethod

class ILayer:

    @abstractmethod
    def forward(self, prev_layer):
        raise NotImplementedError

    @abstractmethod
    def backward(self, prev_layer, leran_rate):
        raise NotImplementedError
