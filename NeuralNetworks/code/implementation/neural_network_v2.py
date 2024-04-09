import numpy as np
import pandas as pd
from layer_v2 import Layer
from optimizers_builder import OptimizersBuilder
from cost_function_builder import CostFunctionBuilder
from helper_functions import HelperFunction


class NeuralNetwork:
    __slots__ = [
        "layers",
        "optimizer",
        "cost_function",
        "layer_sizes",
        "regularization",
        "C",
    ]

    def __init__(
        self, optimizer="adam", cost_function="mse", regularization=None, C=0.01
    ):
        self.layers = []
        self.layer_sizes = []
        self.optimizer = OptimizersBuilder().build_optimizer(optimizer)
        self.cost_function = CostFunctionBuilder().build_cost_function(cost_function)
        assert regularization in [None, "l1", "l2"], "Regularization not supported"
        self.regularization = regularization
        self.C = C  # strength of regularization

    def add_layer(self, layer: Layer):
        if not self.layer_sizes:
            self.layer_sizes = [layer.nodes_in]
        else:
            assert (
                layer.nodes_in == self.layer_sizes[-1]
            ), f"Output in previous layer doesn't match input in this layer"
        self.layer_sizes.append(layer.nodes_out)
        self.layers.append(layer)

    def _forward(self, x: np.ndarray):
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.to_numpy()
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
        a = x.T
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def predict(self, x: np.ndarray):
        return self._forward(x).T

    def predict_class(self, x: np.ndarray):
        return np.argmax(self.predict(x), axis=1, keepdims=True)

    def __str__(self) -> str:
        for i, layer in enumerate(self.layers, 1):
            print(layer.report_layer(i))
        return ""

    def flatten_weights_and_biases(self):
        weights_and_biases = []
        for layer in self.layers:
            weights_and_biases.append(layer.weights.flatten())
            weights_and_biases.append(layer.biases.flatten())
        return np.concatenate(weights_and_biases)

    def deflatten_weights_and_biases(self, solution):
        for layer in self.layers:
            layer.weights = solution[: layer.weights.size].reshape(layer.weights.shape)
            solution = solution[layer.weights.size :]
            layer.biases = solution[: layer.biases.size].reshape(layer.biases.shape)
            solution = solution[layer.biases.size :]

    def flatted_gradient(self):
        gradients = []
        for layer in self.layers:
            gradients.append(layer.weights_gradient.flatten())
            gradients.append(layer.biases_gradient.flatten())
        return np.concatenate(gradients)

    def calculate_gradient_numerically(self, x: np.ndarray, y: np.ndarray, h=1e-5):
        """
        THIS FUNCTION IS ONLY FOR EXPERIMENTAL PURPOSES
        It calculates the gradient numerically.
        It is to slow to be used in practice
        """
        initial_cost = self.cost_function.cost(self.predict(x), y)
        for layer in self.layers:
            for i in range(layer.weights.shape[0]):
                for j in range(layer.weights.shape[1]):
                    layer.weights[i, j] += h
                    new_cost = self.cost_function.cost(self.predict(x), y)
                    layer.weights[i, j] -= h

                    layer.weights_gradient[i, j] = (new_cost - initial_cost) / h

            for i in range(layer.biases.shape[0]):
                for j in range(layer.biases.shape[1]):
                    layer.biases[i, j] += h
                    new_cost = self.cost_function.cost(self.predict(x), y)
                    layer.biases[i, j] -= h
                    layer.biases_gradient[i, j] = (new_cost - initial_cost) / h

    def backpropagation(self, x: np.ndarray, y: np.ndarray):
        a = self._forward(x)
        y = y.T
        delta = self.cost_function.cost_derivative(a, y) * self.layers[
            -1
        ].activation.derivative(self.layers[-1].z)

        # Calculate gradients for the last layer
        self.layers[-1].biases_gradient = np.mean(delta, axis=1, keepdims=True)
        self.layers[-1].weights_gradient = (
            np.dot(delta, self.layers[-2].a.T) / x.shape[0]
        )

        # Add regularization to the last layer
        if self.regularization == "l1":
            self.layers[-1].weights_gradient += (
                self.C * np.sign(self.layers[-1].weights) / x.shape[0]
            )
        elif self.regularization == "l2":
            self.layers[-1].weights_gradient += (
                2 * self.C * self.layers[-1].weights / x.shape[0]
            )

        for previous_layer, layer, next_layer in zip(
            self.layers[-3::-1], self.layers[-2::-1], self.layers[::-1]
        ):
            delta = np.dot(next_layer.weights.T, delta) * layer.activation.derivative(
                layer.z
            )

            # Calculate gradients for the all but first hidden layer
            layer.biases_gradient = np.mean(delta, axis=1, keepdims=True)
            layer.weights_gradient = np.dot(delta, previous_layer.a.T) / x.shape[0]

            # Add regularization to the layer
            if self.regularization == "l1":
                layer.weights_gradient += self.C * np.sign(layer.weights) / x.shape[0]
            elif self.regularization == "l2":
                layer.weights_gradient += 2 * self.C * layer.weights / x.shape[0]

        delta = np.dot(self.layers[1].weights.T, delta) * self.layers[
            0
        ].activation.derivative(self.layers[0].z)

        # Calculate gradients for the first hidden layer
        self.layers[0].biases_gradient = np.mean(delta, axis=1, keepdims=True)
        self.layers[0].weights_gradient = np.dot(delta, x) / x.shape[0]

        # Add regularization to the first hidden layer
        if self.regularization == "l1":
            self.layers[0].weights_gradient += (
                self.C * np.sign(self.layers[0].weights) / x.shape[0]
            )
        elif self.regularization == "l2":
            self.layers[0].weights_gradient += (
                2 * self.C * self.layers[0].weights / x.shape[0]
            )

    def calculate_and_extract_gradient(
        self,
        x: np.ndarray,
        y: np.ndarray,
        current_solution: np.ndarray,
        use_backpropagation=True,
    ):
        self.deflatten_weights_and_biases(current_solution)
        if use_backpropagation:
            self.backpropagation(x, y)
        else:
            self.calculate_gradient_numerically(x, y)
        return self.flatted_gradient()

    def train(
        self,
        X,
        y,
        learning_rate=0.01,
        max_num_epoch=1000,
        batch_size=30,
        batch_fraction=None,
        using_backpropagation=True,
        silent=True,
    ):
        mse_after_epoch_train = self.optimizer(
            X=X,
            y=y,
            using_backpropagation=using_backpropagation,
            learning_rate=learning_rate,
            max_num_epoch=max_num_epoch,
            batch_size=batch_size,
            batch_fraction=batch_fraction,
            neural_network=self,
            silent=silent,
        )
        return mse_after_epoch_train

    def train_with_early_stopping(
        self,
        X,
        y,
        X_test,
        y_test,
        learning_rate=0.01,
        max_num_epoch=1000,
        batch_size=30,
        batch_fraction=None,
        using_backpropagation=True,
        silent=True,
    ):
        mse_after_epoch_train, mse_after_epoch_test = self.optimizer(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            using_backpropagation=using_backpropagation,
            learning_rate=learning_rate,
            max_num_epoch=max_num_epoch,
            batch_size=batch_size,
            batch_fraction=batch_fraction,
            neural_network=self,
            silent=silent,
        )
        return mse_after_epoch_train, mse_after_epoch_test

    def calculate_cost(self, x: np.ndarray, y: np.ndarray):
        base_cost = self.cost_function.cost(self.predict(x), y)
        if self.regularization is None:
            return base_cost
        if self.regularization == "l1":
            return base_cost + self.C * sum(
                np.sum(np.abs(layer.weights)) for layer in self.layers
            )
        if self.regularization == "l2":
            return base_cost + self.C * sum(
                np.sum(layer.weights**2) for layer in self.layers
            )

    def visualize_network(self):
        HelperFunction.visualize_network(self)
