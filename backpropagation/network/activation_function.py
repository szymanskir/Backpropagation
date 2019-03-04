from abc import ABCMeta
import numpy as np


class IActivationFunction(metaclass=ABCMeta):

    def calculate_value(self, x: np.array):
        pass

    def calculate_derivative_value(self, x: np.array):
        pass


class SigmoidActivationFunction(IActivationFunction):

    def calculate_value(self, x: np.array):
        return 1 / (1 + np.exp(-x))

    def calculate_derivative_value(self, x: np.array):
        return self.calculate_value(x) * (1 - self.calculate_value(x))


class IdentityActivationFunction(IActivationFunction):

    def calculate_value(self, x: np.array):
        return x

    def calculate_derivative_value(self, x: np.array):
        return np.ones(x.shape)
