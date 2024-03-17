import numpy as np


class Layer:
    __slots__ = [
        # weights
        "weights",
        "weights_gradient",
        "weights_momentum",
        "weights_gradient_squared",
        "weights_changes",
        # biases
        "biases",
        "biases_gradient",
        "biases_changes",
        "biases_momentum",
        "biases_gradient_squared",
        # activation functions
        "activation",
        "activation_derivative",
        # backpropagation
        "delta",
        "a",
        "z",
    ]

    def __init__(self, nodes_in, nodes_out, activation="sigmoid"):
        self.weights = np.random.normal(size=(nodes_in, nodes_out), scale=1)
        self.biases = np.random.normal(size=(1, nodes_out))
        self.biases_gradient = np.zeros(self.biases.shape)
        self.weights_gradient = np.zeros(self.weights.shape)
        self.biases_changes = np.zeros(self.biases.shape)
        self.weights_changes = np.zeros(self.weights.shape)
        self.biases_momentum = np.zeros(self.biases.shape)
        self.weights_momentum = np.zeros(self.weights.shape)
        self.biases_gradient_squared = np.zeros(self.biases.shape)
        self.weights_gradient_squared = np.zeros(self.weights.shape)

        if activation == "sigmoid":
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == "linear":
            self.activation = self.linear
            self.activation_derivative = self.linear_derivative

    def calculate_layer(self, input):
        """
        Calculate the output of the layer
        Takes in a numpy array and returns a numpy array
        """
        return self.activation(np.dot(input, self.weights) + self.biases)

    def calculate_layer_and_before_activation(self, input):
        """
        Calculate the output of the layer and the output before the activation function
        Takes in a numpy array and returns a numpy array
        """
        return np.dot(input, self.weights) + self.biases, self.activation(
            np.dot(input, self.weights) + self.biases
        )

    def sigmoid(self, x):
        """
        Sigmoid activation function
        Takes in a numpy array and returns a numpy array
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Sigmoid derivative function
        Takes in a numpy array and returns a numpy array
        """
        return x * (1 - x)

    def linear(self, x):
        """
        Linear activation function
        Takes in a numpy array and returns a numpy array
        """
        return x

    def linear_derivative(self, x):
        """
        Linear derivative function
        Takes in a numpy array and returns a numpy array
        """
        return np.ones_like(x)

    def report_layer(self, layer_num):
        return (
            f"Layer number {layer_num}\nWeights\n{self.weights}\nbiases\n{self.biases}"
        )

    def apply_gradient(self, learning_rate):
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.biases_gradient

    def apply_changes(self):
        self.weights += self.weights_changes
        self.biases += self.biases_changes

    def calculate_momentum(self, momentum_decay):
        self.biases_momentum = (
            momentum_decay * self.biases_momentum
            + (1 - momentum_decay) * self.biases_gradient
        )
        self.weights_momentum = (
            momentum_decay * self.weights_momentum
            + (1 - momentum_decay) * self.weights_gradient
        )

    def calculate_momentum_v2(self, momentum_decay):
        self.biases_momentum = (
            momentum_decay * self.biases_momentum + self.biases_gradient
        )
        self.weights_momentum = (
            momentum_decay * self.weights_momentum + self.weights_gradient
        )

    def calculate_gradient_squared(self, decay_rate):
        self.biases_gradient_squared = (
            decay_rate * self.biases_gradient_squared
            + (1 - decay_rate) * self.biases_gradient**2
        )
        self.weights_gradient_squared = (
            decay_rate * self.weights_gradient_squared
            + (1 - decay_rate) * self.weights_gradient**2
        )

    def calculate_gradient_squared_v3(self, decay_rate):
        self.biases_gradient_squared = (
            decay_rate * self.biases_gradient_squared
            + (1 - decay_rate) * self.biases_gradient**2
        )
        self.weights_gradient_squared = (
            decay_rate * self.weights_gradient_squared
            + (1 - decay_rate) * self.weights_gradient**2
        )

    def calculate_changes(self, learning_rate, epsilon, t, momentum_decay, decay_rate):
        self.weights_changes = (
            (
                -learning_rate
                / (
                    np.sqrt(self.weights_gradient_squared / (1 - decay_rate**t))
                    + epsilon
                )
            )
            * self.weights_momentum
            / (1 - momentum_decay**t)
        )

        self.biases_changes = (
            -learning_rate
            / (np.sqrt(self.biases_gradient_squared / (1 - decay_rate**t) + epsilon))
            * self.biases_momentum
            / (1 - momentum_decay**t)
        )

    def calculate_changes_v2(self, learning_rate):
        self.weights_changes = learning_rate * self.weights_momentum
        self.biases_changes = learning_rate * self.biases_momentum

    def calculate_changes_v3(self, learning_rate, epsilon):
        self.weights_changes = (
            -learning_rate
            * self.weights_gradient
            / (np.sqrt(self.weights_gradient_squared) + epsilon)
        )

        self.biases_changes = (
            -learning_rate
            * self.biases_gradient
            / (np.sqrt(self.biases_gradient_squared) + epsilon)
        )
