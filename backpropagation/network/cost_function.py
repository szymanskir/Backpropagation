from abc import ABCMeta
from typing import List
import numpy as np


class ICostFunction(metaclass=ABCMeta):
    def calculate_value(self, y: np.array, a: np.array):
        pass

    def calculate_derivative_value(self, y: np.array, a: np.array):
        pass


class MSECostFunction(ICostFunction):
    def calculate_value(self, y: List[np.array], a: List[np.array]) -> float:
        n = len(y)

        costs = [np.power(np.linalg.norm(expected - output), 2)
                 for expected, output in zip(y, a)]

        return sum(costs) / (2 * n)

    def calculate_derivative_value(self, y: np.array, a: np.array) -> np.array:
        return a - y


class SECostFunction(ICostFunction):
    def calculate_value(self, y: List[np.array], a: List[np.array]) -> float:
        cost = [np.power(expected - output, 2)
                for expected, output in zip(y, a)]
        return sum(cost) / 2

    def calculate_derivative_value(self, y: np.array, a: np.array) -> np.array:
        return a - y
