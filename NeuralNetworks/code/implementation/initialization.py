from abc import ABC, abstractmethod
import numpy as np


class Initialization(ABC):
    @abstractmethod
    def initialize(self, n_input: int, n_output: int):
        pass


class NormalXavier(Initialization):
    def initialize(self, n_input: int, n_output: int):
        std = np.sqrt(2 / (n_input + n_output))
        return np.random.normal(0, std, (n_input, n_output))


class UniformXavier(Initialization):
    def initialize(self, n_input: int, n_output: int):
        val = np.sqrt(6 / (n_input + n_output))
        return np.random.uniform(-val, val, (n_input, n_output))


class He(Initialization):
    def initialize(self, n_input: int, n_output: int):
        return np.random.normal(0, np.sqrt(2 / n_input), (n_input, n_output))


class Zero(Initialization):
    def initialize(self, n_input: int, n_output: int):
        return np.zeros((n_input, n_output))


class Normal(Initialization):
    def initialize(self, n_input: int, n_output: int):
        return np.random.normal(0, 1, (n_input, n_output))


class Uniform_minus_one_one(Initialization):
    def initialize(self, n_input: int, n_output: int):
        return np.random.uniform(-1, 1, (n_input, n_output))


class Uniform_zero_one(Initialization):
    def initialize(self, n_input: int, n_output: int):
        return np.random.uniform(0, 1, (n_input, n_output))
