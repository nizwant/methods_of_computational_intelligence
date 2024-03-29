import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from abc import abstractmethod


class HelperFunction:

    @abstractmethod
    def visualize_network(network):
        """
        Visualize the network architecture
        """

        G = nx.complete_multipartite_graph(*network.layer_sizes)
        G.remove_edges_from(G.edges())

        # relabeling the nodes and adding biases
        counter = 0
        mapping = {}
        biases = {}
        for layer_number, layer_size in enumerate(network.layer_sizes):
            for j in range(layer_size):
                if layer_number == 0:
                    biases[counter] = 0
                else:
                    biases[counter] = network.layers[layer_number - 1].biases[j, 0]
                mapping[counter] = f"({layer_number}, {j})"
                counter += 1
        nx.set_node_attributes(G, biases, "bias")
        G = nx.relabel_nodes(G, mapping)

        # add edges
        for layer_number, layer_size in enumerate(network.layer_sizes[:-1]):
            for j in range(layer_size):
                for k in range(network.layer_sizes[layer_number + 1]):
                    weight = network.layers[layer_number].weights[k, j]
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
        if max_bias == 0:
            max_bias = 1
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
