from initialization import He
from initialization import Normal
from initialization import NormalXavier
from initialization import UniformXavier
from initialization import Uniform_minus_one_one
from initialization import Uniform_zero_one
from initialization import Zero


class InitializationBuilder:
    @staticmethod
    def get_initialization(initialization: str, nodes_out: int, nodes_in: int):
        if initialization == "he":
            return He().initialize(nodes_out, nodes_in)
        elif initialization == "normal":
            return Normal().initialize(nodes_out, nodes_in)
        elif initialization == "normal_xavier":
            return NormalXavier().initialize(nodes_out, nodes_in)
        elif initialization == "uniform_xavier":
            return UniformXavier().initialize(nodes_out, nodes_in)
        elif initialization == "uniform_minus_one_one":
            return Uniform_minus_one_one().initialize(nodes_out, nodes_in)
        elif initialization == "uniform_zero_one":
            return Uniform_zero_one().initialize(nodes_out, nodes_in)
        elif initialization == "zero":
            return Zero().initialize(nodes_out, nodes_in)
        else:
            raise ValueError(f"Initialization function {initialization} not supported")
