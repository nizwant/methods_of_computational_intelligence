import numpy as np


class NeuralNetwork:

    def init(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [
            np.random.rand(self.layer_sizes[i], self.layer_sizes[i + 1])
            for i in range(len(self.layer_sizes) - 1)
        ]
        self.biases = [np.random.rand(size) for size in self.layer_sizes[1:]]

    def feedforward(self, inputs, activation_function):
        for i in range(len(self.layer_sizes) - 1):
            inputs = np.dot(inputs, self.weights[i]) + self.biases[i]
            inputs = activation_function(inputs)
        return inputs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def identity(self, x):
        return x
