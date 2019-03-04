from abc import ABCMeta
import numpy as np


class ICostFunction(metaclass=ABCMeta):
    def calculate_value(self, y: np.array, a: np.array):
        pass

    def calculate_derivative_value(self, y: np.array, a: np.array):
        pass


class MSECostFunction(ICostFunction):
    def calculate_value(self, y: np.array, a: np.array) -> int:
        n = y.shape[0]
        return np.sum(np.power(np.linalg.norm(y - a), 2)) / (2 * n)

    def calculate_derivative_value(self, y: np.array, a: np.array) -> np.array:
        return a - y


class SECostFunction(ICostFunction):
    def calculate_value(self, y: np.array, a: np.array) -> int:
        return np.sum(np.power(y - a, 2)) / 2

    def calculate_derivative_value(self, y: np.array, a: np.array) -> np.array:
        return a - y
