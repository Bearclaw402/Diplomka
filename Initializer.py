import numpy
from scipy.stats import gennorm


class Initializer:
    def __init__(self, n_in, n_out, shape, initializer='xavier_uniform'):
        self.n_in = n_in
        self.n_out = n_out
        self.shape = shape
        self.initializer = initializer

    def initializeWeights(self, mean=0.0, dev=0.0, low=0.0, high=0.0, df=1.0, beta=1.0):
        initializer = self.initializer.split('_')[0]
        distribution = self.initializer.split('_')[1]
        if initializer == 'xavier':
            if 'norm' in distribution:
                dev = numpy.sqrt(2.0 / (self.n_in + self.n_out))
            elif distribution == 'uniform':
                low = -numpy.sqrt(6.0 / (self.n_in + self.n_out))
                high = numpy.sqrt(6.0 / (self.n_in + self.n_out))
            else:
                dev = numpy.sqrt(1.0 / (self.n_in + self.n_out))
        elif initializer == 'he':
            if 'norm' in distribution:
                dev = numpy.sqrt(2.0 / self.n_in)
            elif distribution == 'uniform':
                low = -numpy.sqrt(6.0 / self.n_in)
                high = numpy.sqrt(6.0 / self.n_in)
            else:
                dev = numpy.sqrt(1.0 / self.n_in)
        elif initializer == 'random':
            low = -0.05
            high = 0.05
        else:
            dev = 1. / self.n_in

        if distribution == 'normal':
            weights = self.__normal__(self.shape, mean, dev)
        elif distribution == 'uniform':
            weights = self.__uniform__(self.shape, low, high)
        elif distribution == 'student':
            weights = dev * self.__student__(self.shape, df)
        elif distribution == 'chisqr':
            weights = dev * self.__chisqr__(self.shape, df)
        elif distribution == 'gennorm':
            weights = self.__gennorm__(self.shape, beta, mean, dev)
        elif distribution == 'lognorm':
            weights = self.__lognorm__(self.shape, mean, dev)
        else:
            weights = self.__normal__(self.shape)
        return weights

    @staticmethod
    def __normal__(shape, mean=0.0, dev=0.05):
        return numpy.random.normal(loc=mean, scale=dev, size=shape)

    @staticmethod
    def __uniform__(shape, low=-0.05, high=0.05):
        return numpy.random.uniform(low=low, high=high, size=shape)

    @staticmethod
    def __student__(shape, df=1.0):
        return numpy.random.standard_t(df=df, size=shape)

    @staticmethod
    def __chisqr__(shape, df=1.0):
        return numpy.random.chisquare(df=df, size=shape)

    @staticmethod
    def __gennorm__(shape, beta=1.0, mean=0.0, dev=0.05):
        return gennorm.rvs(beta=beta, loc=mean, scale=dev, size=shape)

    @staticmethod
    def __lognorm__(shape, mean=0.0, dev=0.05):
        return numpy.random.lognormal(mean=mean, sigma=dev, size=shape)