import numpy as np
from initialization_bulider import InitializationBuilder
from activation_bulider import ActivationBuilder


class Layer:
    __slots__ = [
        "nodes_in",
        "nodes_out",
        "weights",
        "weights_gradient",
        "biases",
        "biases_gradient",
        "activation",
        "a",
        "z",
    ]

    def __init__(
        self,
        nodes_in,
        nodes_out,
        activation="sigmoid",
        weight_initialization="he",
        bias_initialization="zero",
    ):

        self.nodes_in = nodes_in
        self.nodes_out = nodes_out

        self.weights = InitializationBuilder.get_initialization(
            weight_initialization, nodes_in, nodes_out
        )

        self.weights_gradient = InitializationBuilder.get_initialization(
            "zero", nodes_in, nodes_out
        )

        self.biases = InitializationBuilder.get_initialization(
            bias_initialization, 1, nodes_out
        )
        self.biases_gradient = InitializationBuilder.get_initialization(
            "zero", 1, nodes_out
        )

        self.activation = ActivationBuilder.get_activation(activation)

    def forward(self, a):
        self.z = np.dot(a, self.weights) + self.biases
        self.a = self.activation.activation_function(self.z)
        return self.a
