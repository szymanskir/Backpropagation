from abc import ABCMeta
from typing import List
import numpy as np


class ICostFunction(metaclass=ABCMeta):
    def calculate_value(self, y: np.array, a: np.array):
        pass

    def calculate_derivative_value(self, y: np.array, a: np.array):
        pass


class MSECostFunction(ICostFunction):
    def calculate_value(self, y: np.array, a: np.array) -> float:
        n = y.shape[1]

        costs = np.power(np.linalg.norm(y - a, axis=0), 2)
        return sum(costs) / (2 * n)

    def calculate_derivative_value(self, y: np.array, a: np.array) -> np.array:
        return a - y


class SECostFunction(ICostFunction):
    def calculate_value(self, y: np.array, a: np.array) -> float:
        costs = np.power(y - a, 2)
        return np.sum(costs) / 2

    def calculate_derivative_value(self, y: np.array, a: np.array) -> np.array:
        return a - y


class CrossEntropyCostFunction(ICostFunction):
    def calculate_value(self, y: List[np.array], a: List[np.array]) -> float:
        return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

    def calculate_derivative_value(self, y: np.array, a: np.array) -> np.array:
        return a - y
