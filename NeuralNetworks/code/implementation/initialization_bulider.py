from initialization import He
from initialization import Normal
from initialization import NormalXavier
from initialization import UniformXavier
from initialization import Uniform_minus_one_one
from initialization import Uniform_zero_one
from initialization import Zero


class InitializationBuilder:
    @staticmethod
    def get_initialization(initialization: str, nodes_in: int, nodes_out: int):
        if initialization == "he":
            return He().initialize(nodes_in, nodes_out)
        elif initialization == "normal":
            return Normal().initialize(nodes_in, nodes_out)
        elif initialization == "normal_xavier":
            return NormalXavier().initialize(nodes_in, nodes_out)
        elif initialization == "uniform_xavier":
            return UniformXavier().initialize(nodes_in, nodes_out)
        elif initialization == "uniform_minus_one_one":
            return Uniform_minus_one_one().initialize(nodes_in, nodes_out)
        elif initialization == "uniform_zero_one":
            return Uniform_zero_one().initialize(nodes_in, nodes_out)
        elif initialization == "zero":
            return Zero().initialize(nodes_in, nodes_out)
        else:
            raise ValueError(f"Initialization function {initialization} not supported")
