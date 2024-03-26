import numpy as np
from abc import ABC, abstractmethod
import pandas as pd


class CostFunction(ABC):
    @abstractmethod
    def cost(self, y, y_hat):
        pass

    @abstractmethod
    def cost_derivative(self, y, y_hat):
        pass


class MeanSquaredError(CostFunction):
    def cost(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.to_numpy()
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return np.mean((y - y_hat) ** 2)

    def cost_derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * (y_hat - y)


class AbsoluteError(CostFunction):
    def cost(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.to_numpy()
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return np.mean(np.abs(y - y_hat))

    def cost_derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sign(y_hat - y)


class CrossEntropyWithSoftmax(CostFunction):
    EPSILON = 1e-10

    def cost(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        y_hat = np.clip(y_hat, self.EPSILON, 1)
        return -np.mean(y * np.log(y_hat))

    def cost_derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y_hat - y


# class CrossEntropy(CostFunction):
#     def cost(self, y_hat: np.ndarray, y: np.ndarray) -> float:
#         return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

#     def cost_derivative(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
#         return (y_hat - y) / (y_hat * (1 - y_hat))
