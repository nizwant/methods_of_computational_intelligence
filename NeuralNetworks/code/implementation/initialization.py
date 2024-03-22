from abc import ABC, abstractmethod
import numpy as np


class Initialization(ABC):
    @abstractmethod
    def initialize(self, n_output: int, n_input: int):
        pass


class NormalXavier(Initialization):
    def initialize(self, n_output: int, n_input: int):
        std = np.sqrt(2 / (n_output + n_input))
        return np.random.normal(0, std, (n_output, n_input))


class UniformXavier(Initialization):
    def initialize(self, n_output: int, n_input: int):
        val = np.sqrt(6 / (n_output + n_input))
        return np.random.uniform(-val, val, (n_output, n_input))


class He(Initialization):
    def initialize(self, n_output: int, n_input: int):
        return np.random.normal(0, np.sqrt(2 / n_output), (n_output, n_input))


class Zero(Initialization):
    def initialize(self, n_output: int, n_input: int):
        return np.zeros((n_output, n_input))


class Normal(Initialization):
    def initialize(self, n_output: int, n_input: int):
        return np.random.normal(0, 1, (n_output, n_input))


class Uniform_minus_one_one(Initialization):
    def initialize(self, n_output: int, n_input: int):
        return np.random.uniform(-1, 1, (n_output, n_input))


class Uniform_zero_one(Initialization):
    def initialize(self, n_output: int, n_input: int):
        return np.random.uniform(0, 1, (n_output, n_input))
