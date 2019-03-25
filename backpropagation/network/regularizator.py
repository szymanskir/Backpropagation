from abc import ABCMeta
from typing import List

import numpy as np


class IRegulatizator(metaclass=ABCMeta):
    def get_term(self, regularization_param: float, weights: List[np.array], n: int) -> float:
        pass

    def get_derivative_terms(
            self,
            regularization_param: float,
            weights: List[np.array],
            n: int
    ) -> List[np.array]:
        pass


class L1Regularizator(IRegulatizator):
    def get_term(self, regularization_param: float, weights: List[np.array], n: int) -> float:
        return (regularization_param / n) * sum([np.sum(np.abs(w)) for w in weights])

    def get_derivative_terms(
            self,
            regularization_param: float,
            weights: List[np.array],
            n: int
    ) -> List[np.array]:
        return [(regularization_param / n) * np.sign(w) for w in weights]


class L2Regularizator(IRegulatizator):
    def get_term(self, regularization_param: float, weights: List[np.array], n: int) -> float:
        return (regularization_param / (2*n)) * sum([np.sum(np.power(w, 2)) for w in weights])

    def get_derivative_terms(
            self,
            regularization_param: float,
            weights: List[np.array],
            n: int
    ) -> List[np.array]:
        return [(regularization_param / n) * w for w in weights]
