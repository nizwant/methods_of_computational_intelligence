import numpy as np
from abc import ABC, abstractmethod


class CostFunction(ABC):
    @abstractmethod
    def cost(self, y, y_hat):
        pass

    @abstractmethod
    def cost_derivative(self, y, y_hat):
        pass


class MeanSquaredError(CostFunction):
    def cost(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return np.mean((y - y_hat) ** 2)

    def cost_derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * (y_hat - y)


class AbsoluteError(CostFunction):
    def cost(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return np.mean(np.abs(y - y_hat))

    def cost_derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sign(y_hat - y)


class CrossEntropyWithSoftmax(CostFunction):
    def cost(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        return -np.mean(y * np.log(y_hat))

    def cost_derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y_hat - y


# class CrossEntropy(CostFunction):
#     def cost(self, y_hat: np.ndarray, y: np.ndarray) -> float:
#         return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

#     def cost_derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
#         return (y_hat - y) / (y_hat * (1 - y_hat))
