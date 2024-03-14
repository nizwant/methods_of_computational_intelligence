from initialization import He
from initialization import Normal
from initialization import NormalXavier
from initialization import UniformXavier
from initialization import Uniform_minus_one_one
from initialization import Uniform_zero_one
from initialization import Zero


class InitializationBuilder:
    @staticmethod
    def get_initialization(initialization: str):
        if initialization == "he":
            return He()
        elif initialization == "normal":
            return Normal()
        elif initialization == "normal_xavier":
            return NormalXavier()
        elif initialization == "uniform_xavier":
            return UniformXavier()
        elif initialization == "uniform_minus_one_one":
            return Uniform_minus_one_one()
        elif initialization == "uniform_zero_one":
            return Uniform_zero_one()
        elif initialization == "zero":
            return Zero()
        else:
            raise ValueError(f"Initialization function {initialization} not supported")
