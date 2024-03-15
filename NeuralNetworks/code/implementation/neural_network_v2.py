import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from .layer import Layer
from .optimizers_bulider import OptimizersBuilder


class NeuralNetwork:
    __slots__ = ["layers", "optimizer", "cost_function", "layer_sizes"]

    def __init__(self, optimizer="adam", cost_function="mse"):
        self.layers = []
        self.optimizer = OptimizersBuilder().build_optimizer(optimizer)
        self.cost_function = None
        self.layer_sizes = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        if self.layer_sizes is None:
            self.layer_sizes = [layer.nodes_in]
        self.layer_sizes.append(layer.nodes_out)
