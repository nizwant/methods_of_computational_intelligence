import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from layer_v2 import Layer
from optimizers_bulider import OptimizersBuilder
from cost_funtion_builder import CostFunctionBuilder


class NeuralNetwork:
    __slots__ = ["layers", "optimizer", "cost_function", "layer_sizes"]

    def __init__(self, optimizer="adam", cost_function="mse"):
        self.layers = []
        self.optimizer = OptimizersBuilder().build_optimizer(optimizer)
        self.cost_function = CostFunctionBuilder().build_cost_function(cost_function)
        self.layer_sizes = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        if not self.layer_sizes:
            self.layer_sizes = [layer.nodes_in]
        self.layer_sizes.append(layer.nodes_out)

    def forward(self, x: np.ndarray):
        if x is pd.DataFrame:
            x = x.to_numpy()
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

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
                if layer_number == 0:
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
        plt.show()

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

    def backpropagation(self, x: np.ndarray, y: np.ndarray):
        a = self.forward(x)
        delta = (
            self.cost_function.cost_derivative(a, y)
            * self.layers[-1].activation.derivative(self.layers[-1].z).T
        )
        for previous_layer, layer in zip(self.layers[-2::-1], self.layers[::-1]):
            layer.biases_gradient = np.mean(delta, axis=1)
            layer.weights_gradient = np.dot(delta, previous_layer.a) / x.shape[0]
            delta = np.dot(delta.T, layer.weights.T) * layer.activation.derivative(
                layer.z
            )


nn = NeuralNetwork()

np.random.seed(0)

nn.add_layer(
    Layer(
        1,
        2,
        activation="sigmoid",
        weight_initialization="normal",
        bias_initialization="normal",
    )
)
nn.add_layer(
    Layer(
        2,
        1,
        activation="linear",
        weight_initialization="normal",
        bias_initialization="normal",
    )
)

print(nn.flatted_gradient())
nn.backpropagation(np.array([[2], [1], [3]]), np.array([[1], [2], [4]]))
print(nn.flatted_gradient())
