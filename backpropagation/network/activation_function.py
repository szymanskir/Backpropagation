from abc import ABCMeta
import numpy as np


class IActivationFunction(metaclass=ABCMeta):
    """Interface representing an activation function for neural networks.
    """

    def calculate_value(self, x: np.array) -> np.array:
        """Return the value of the function for values in x.
        """
        pass

    def calculate_derivative_value(self, x: np.array) -> np.array:
        """Return the value of the derivative for values in x.
        """
        pass


class IdentityActivationFunction(IActivationFunction):

    def calculate_value(self, x: np.array) -> np.array:
        return x

    def calculate_derivative_value(self, x: np.array) -> np.array:
        return np.ones(x.shape)


class SigmoidActivationFunction(IActivationFunction):

    def calculate_value(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def calculate_derivative_value(self, x: np.array) -> np.array:
        return self.calculate_value(x) * (1 - self.calculate_value(x))


class ReLUActivationFunction(IActivationFunction):

    def calculate_value(self, x: np.array) -> np.array:
        return x.clip(0)

    def calculate_derivative_value(self, x: np.array) -> np.array:
        return np.where(x < 0, 0, 1)


class SoftmaxActivationFunction(IActivationFunction):

    def calculate_value(self, x: np.array) -> np.array:
        return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

    def calculate_derivative_value(self, x: np.array) -> np.array:
        values = self.calculate_value(x)
        return values * (1 - values)
