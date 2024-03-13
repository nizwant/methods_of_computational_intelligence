from activation import Sigmoid
from activation import ReLU
from activation import Tanh
from activation import Softmax
from activation import Linear
from activation import LeakyReLU


class ActivationBuilder:
    @staticmethod
    def get_activation(activation: str):
        if activation == "sigmoid":
            return Sigmoid()
        elif activation == "relu":
            return ReLU()
        elif activation == "tanh":
            return Tanh()
        elif activation == "softmax":
            return Softmax()
        elif activation == "linear":
            return Linear()
        elif activation == "leaky_relu":
            return LeakyReLU()
        else:
            raise ValueError(f"Activation function {activation} not supported")
