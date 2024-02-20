import numpy as np


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs, activation_function):
        total = np.dot(self.weights, inputs) + self.bias
        return activation_function(total)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
