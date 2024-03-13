from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def activation_function(self, x: float):
        pass

    @abstractmethod
    def activation_function_derivative(self, x: float):
        pass


class Sigmoid(Activation):
    def activation_function(self, x: float):
        return 1 / (1 + np.exp(-x))

    def activation_function_derivative(self, x: float):
        return self.activation_function(x) * (1 - self.activation_function(x))


class ReLU(Activation):
    def activation_function(self, x: float):
        return max(0, x)

    def activation_function_derivative(self, x: float):
        return 0 if x < 0 else 1


class Tanh(Activation):
    def activation_function(self, x: float):
        return np.tanh(x)

    def activation_function_derivative(self, x: float):
        return 1 - np.tanh(x) ** 2


class Softmax(Activation):
    """
    Softmax function is used in the output layer of a neural network for multi-class classification problems.
    It squashes the output of each unit to be between 0 and 1, just like a sigmoid function.
    It also divides each output such that the total sum of the outputs is equal to 1.
    """

    def activation_function(self, x: float):
        exps = np.exp(x - x.max())
        return exps / np.sum(exps)

    def activation_function_derivative(self, x: float):
        return self.activation_function(x) * (1 - self.activation_function(x))


class Linear(Activation):
    def activation_function(self, x: float):
        return x

    def activation_function_derivative(self, x: float):
        return 1


class LeakyReLU(Activation):
    def activation_function(self, x: float):
        return max(0.01 * x, x)

    def activation_function_derivative(self, x: float):
        return 0.01 if x < 0 else 1
