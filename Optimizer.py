import numpy


class AdamOptimizer:
    def __init__(self, weights, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.theta = weights

    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.theta = self.theta - self.alpha * (m_hat / (numpy.sqrt(v_hat) - self.epsilon))
        return self.theta


class SGD:
    def __init__(self, weights, learn_rate=0.001):
        self.learn_rate = learn_rate
        self.weights = weights

    def backward_pass(self, gradient):
        self.weights -= gradient * self.learn_rate
        return self.weights
