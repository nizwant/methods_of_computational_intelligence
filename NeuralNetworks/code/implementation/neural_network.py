import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from layer import Layer


class NeuralNetwork:
    __slots__ = ["hidden_layers", "layers", "layer_sizes"]

    def __init__(self, hidden_layers, input_size, output_size):
        self.hidden_layers = hidden_layers
        self.layers = []
        self.layer_sizes = [input_size] + hidden_layers + [output_size]

        # Create the input layer
        input_layer = Layer(1, hidden_layers[0])
        self.layers.append(input_layer)

        # Create the hidden layers
        for input_size, output_size in zip(hidden_layers, hidden_layers[1:]):
            self.layers.append(Layer(input_size, output_size))

        # Create the output layer
        output_layer = Layer(hidden_layers[-1], 1, activation="linear")
        self.layers.append(output_layer)

    def forward(self, input):
        """
        Takes a input and returns the output of the network
        """
        for layer in self.layers:
            input = layer.calculate_layer(input)
        return input

    def backpropagation(self, input, output):
        """
        Perform backpropagation on the network
        """

        for x, y in zip(input, output):
            # Forward pass
            a = x
            for layer in self.layers:
                z, a = layer.calculate_layer_and_before_activation(a)
                layer.z = z
                layer.a = a

            # Backward pass
            # Calculate the delta for the output layer
            self.layers[-1].delta = self.mean_squared_error_gradient(
                self.layers[-1].a, y
            ) * self.layers[-1].activation_derivative(self.layers[-1].z)

            # Calculate the delta for the hidden layers
            for layer, next_layer in zip(self.layers[-2::-1], self.layers[-1::-1]):
                layer.delta = np.dot(
                    next_layer.delta, next_layer.weights.T
                ) * layer.activation_derivative(layer.a)

            # Calculate the gradients for the input layer
            self.layers[0].biases_gradient += self.layers[0].delta
            try:
                self.layers[0].weights_gradient += np.dot(x.T, self.layers[0].delta)
            except:
                self.layers[0].weights_gradient += np.dot(x, self.layers[0].delta)

            # Calculate the gradients for the hidden layers
            for previous_layer, layer in zip(self.layers, self.layers[1:]):
                layer.biases_gradient += layer.delta
                layer.weights_gradient += np.dot(previous_layer.a.T, layer.delta)
                # print(
                #     f"previous a {previous_layer.a.T} * delta {layer.delta} = {np.dot(previous_layer.a.T, layer.delta)}"
                # )
                # print()

        for layer in self.layers:
            layer.biases_gradient = layer.biases_gradient / len(input)
            layer.weights_gradient = layer.weights_gradient / len(input)

    def mean_squared_error(self, input, output):
        """
        Calculate the mean squared error of the network on a given dataset and output
        """
        mse = []
        for i, j in zip(input, output):
            mse.append((j - self.forward(i)) ** 2)
        return np.mean(mse)

    def mean_squared_error_gradient(self, predicted, true):
        """
        Calculate the gradient of the mean squared error
        """
        return 2 * (predicted - true)

    def visualize(self):
        """
        Visualize the network architecture
        """
        for i, layer in enumerate(self.layers):
            print(layer.report_layer(i))
            print("\n")

    def apply_gradient(self, learning_rate):
        for layer in self.layers:
            layer.apply_gradient(learning_rate)

    def apply_changes(self):
        for layer in self.layers:
            layer.apply_changes()

    def calculate_momentum(self, momentum_decay):
        """adam"""
        for layer in self.layers:
            layer.calculate_momentum(momentum_decay)

    def calculate_momentum_v2(self, momentum_decay):
        """momentum"""
        for layer in self.layers:
            layer.calculate_momentum_v2(momentum_decay)

    def calculate_gradient_squared(self, decay_rate):
        """adam"""
        for layer in self.layers:
            layer.calculate_gradient_squared(decay_rate)

    def calculate_changes(self, learning_rate, epsilon, t, momentum_decay, decay_rate):
        """adam"""
        for layer in self.layers:
            layer.calculate_changes(
                learning_rate, epsilon, t, momentum_decay, decay_rate
            )

    def calculate_changes_v2(self, learning_rate):
        """momentum"""
        for layer in self.layers:
            layer.calculate_changes_v2(learning_rate)

    def calculate_gradient_squared_v3(self, decay_rate):
        """rmsprop"""
        for layer in self.layers:
            layer.calculate_gradient_squared_v3(decay_rate)

    def calculate_changes_v3(self, learning_rate, epsilon):
        """rmsprop"""
        for layer in self.layers:
            layer.calculate_changes_v3(learning_rate, epsilon)

    def calculate_gradient(self, input, output):
        """
        Calculate the gradient of the network
        """
        h = 0.0000001
        original_mse = self.mean_squared_error(input, output)

        for layer in self.layers:
            for i in range(layer.weights.shape[0]):
                for j in range(layer.weights.shape[1]):
                    layer.weights[i, j] += h
                    new_mse = self.mean_squared_error(input, output)
                    layer.weights_gradient[i, j] = (new_mse - original_mse) / h
                    layer.weights[i, j] -= h

            for i in range(layer.biases.shape[0]):
                for j in range(layer.biases.shape[1]):
                    layer.biases[i, j] += h
                    new_mse = self.mean_squared_error(input, output)
                    layer.biases_gradient[i, j] = (new_mse - original_mse) / h
                    layer.biases[i, j] -= h

    def adam(
        self,
        X,
        y,
        learning_rate=0.01,
        momentum_decay=0.9,
        squared_gradient_decay=0.999,
        max_num_epoch=1000,
        batch_size=1,
        batch_fraction=None,
        epsilon=1e-8,
        silent=False,
    ):

        # initialization
        if type(X) is pd.DataFrame:
            X = X.to_numpy()
        if type(y) is pd.DataFrame:
            y = y.to_numpy().T
        counter = 0

        # set batch size
        assert type(batch_size) is int, "batch_size must be an integer"
        if batch_fraction is not None:
            assert 0 < batch_fraction <= 1, "batch_fraction must be between 0 and 1"
            batch_size = int(X.shape[0] * batch_fraction)
        iterations = int(X.shape[0] / batch_size)
        mse_list = []
        mse_after_epoch = []

        for i in range(max_num_epoch):
            N = X.shape[0]
            shuffled_idx = np.random.permutation(N)
            X, y = X[shuffled_idx], y[shuffled_idx]
            for idx in range(iterations):
                X_selected, y_selected = (
                    X[idx * batch_size : (idx + 1) * batch_size],
                    y[idx * batch_size : (idx + 1) * batch_size],
                )
                self.backpropagation(X_selected, y_selected)

                self.calculate_momentum(momentum_decay)
                self.calculate_gradient_squared(squared_gradient_decay)
                counter += 1

                self.calculate_changes(
                    learning_rate,
                    epsilon,
                    counter,
                    momentum_decay,
                    squared_gradient_decay,
                )
                self.apply_changes()
                mse_list.append(self.mean_squared_error(X_selected, y_selected))
            mse_after_epoch.append(self.mean_squared_error(X, y))
            if not silent:
                print("Epoch:", i)
        return mse_list, mse_after_epoch

    def mini_batch_gradient_descent_with_momentum(
        self,
        X,
        y,
        learning_rate=0.01,
        momentum_decay=0.9,
        max_num_epoch=1000,
        batch_size=1,
        batch_fraction=None,
        silent=False,
    ):
        # initialization
        if type(X) is pd.DataFrame:
            X = X.to_numpy()
        if type(y) is pd.DataFrame:
            y = y.to_numpy().T

        # set batch size
        assert type(batch_size) is int, "batch_size must be an integer"
        if batch_fraction is not None:
            assert 0 < batch_fraction <= 1, "batch_fraction must be between 0 and 1"
            batch_size = int(X.shape[0] * batch_fraction)
        iterations = int(X.shape[0] / batch_size)
        mse_list = []
        mse_after_epoch = []

        for i in range(max_num_epoch):
            N = X.shape[0]
            shuffled_idx = np.random.permutation(N)
            X, y = X[shuffled_idx], y[shuffled_idx]
            for idx in range(iterations):
                X_selected, y_selected = (
                    X[idx * batch_size : (idx + 1) * batch_size],
                    y[idx * batch_size : (idx + 1) * batch_size],
                )
                self.backpropagation(X_selected, y_selected)
                self.calculate_momentum_v2(momentum_decay)
                self.calculate_changes_v2(learning_rate)
                self.apply_changes()
                mse_list.append(self.mean_squared_error(X_selected, y_selected))
            mse_after_epoch.append(self.mean_squared_error(X, y))
            if not silent:
                print("Epoch:", i)
        return mse_list, mse_after_epoch

    def rmsprop(
        self,
        X,
        y,
        learning_rate=0.01,
        squared_gradient_decay=0.999,
        max_num_epoch=1000,
        batch_size=1,
        batch_fraction=None,
        epsilon=1e-8,
        silent=False,
    ):

        # initialization
        if type(X) is pd.DataFrame:
            X = X.to_numpy()
        if type(y) is pd.DataFrame:
            y = y.to_numpy().T

        # set batch size
        assert type(batch_size) is int, "batch_size must be an integer"
        if batch_fraction is not None:
            assert 0 < batch_fraction <= 1, "batch_fraction must be between 0 and 1"
            batch_size = int(X.shape[0] * batch_fraction)
        iterations = int(X.shape[0] / batch_size)
        mse_list = []
        mse_after_epoch = []

        for i in range(max_num_epoch):
            N = X.shape[0]
            shuffled_idx = np.random.permutation(N)
            X, y = X[shuffled_idx], y[shuffled_idx]
            for idx in range(iterations):
                X_selected, y_selected = (
                    X[idx * batch_size : (idx + 1) * batch_size],
                    y[idx * batch_size : (idx + 1) * batch_size],
                )
                self.backpropagation(X_selected, y_selected)
                self.calculate_gradient_squared_v3(squared_gradient_decay)

                self.calculate_changes_v3(learning_rate, epsilon)
                self.apply_changes()
                mse_list.append(self.mean_squared_error(X_selected, y_selected))
            mse_after_epoch.append(self.mean_squared_error(X, y))
            if not silent:
                print("Epoch:", i)
        return mse_list, mse_after_epoch

    def visualize_network(self):
        """
        Visualize the network architecture
        """

        G = nx.complete_multipartite_graph(*self.layer_sizes)
        G.remove_edges_from(G.edges())

        # relabeling the nodes and adding biases
        counter = 0
        mapping = {}
        biases = {}
        for layer_number, layer_size in enumerate(self.layer_sizes):
            for j in range(layer_size):
                if counter == 0:
                    biases[counter] = 0
                else:
                    biases[counter] = self.layers[layer_number - 1].biases[0, j]
                mapping[counter] = f"({layer_number}, {j})"
                counter += 1
        nx.set_node_attributes(G, biases, "bias")
        G = nx.relabel_nodes(G, mapping)

        # add edges
        for layer_number, layer_size in enumerate(self.layer_sizes[:-1]):
            for j in range(layer_size):
                for k in range(self.layer_sizes[layer_number + 1]):
                    weight = self.layers[layer_number].weights[j, k]
                    color = "red" if weight < 0 else "green"
                    G.add_edge(
                        f"({layer_number}, {j})",
                        f"({layer_number+1}, {k})",
                        weight=weight,
                        color=color,
                    )

        # colors and widths of the edges
        edges = G.edges()
        colors = [G[u][v]["color"] for u, v in edges]
        weights = [G[u][v]["weight"] for u, v in edges]

        # colors of the nodes
        node_colors = []
        for _, value in nx.get_node_attributes(G, "bias").items():
            node_colors.append(value)
        cmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "g"], N=256)
        max_bias = max(abs(np.array(node_colors)))

        # draw the graph
        pos = nx.multipartite_layout(G)
        plt.figure(figsize=(5, 5))
        nx.draw(
            G,
            pos,
            node_size=600,
            font_size=10,
            font_weight="bold",
            edgecolors="black",
            linewidths=2,
            edge_color=colors,
            width=weights,
            node_color=node_colors,
            cmap=cmap,
            vmin=-max_bias,
            vmax=max_bias,
        )
        # create legend that says red is negative and green is positive
        red_patch = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Negative",
            markerfacecolor="r",
            markersize=10,
        )
        green_patch = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Positive",
            markerfacecolor="g",
            markersize=10,
        )
        plt.legend(handles=[red_patch, green_patch])


def main():
    np.random.seed(0)
    neural = NeuralNetwork([12], 1, 1)
    df = pd.read_csv(
        "https://raw.githubusercontent.com/nizwant/miowid/main/data/regression/square-small-test.csv"
    )

    neural.backpropagation(df[["x"]].to_numpy(), df[["y"]].to_numpy())
    for layer in neural.layers:
        print(layer.weights_gradient)
        print(layer.biases_gradient)
        print("\n")

    # print(neural.layers[-1].delta)


if __name__ == "__main__":
    main()
