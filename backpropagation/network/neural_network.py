from typing import List, Tuple
import numpy as np

from .activation_function import IActivationFunction
from .cost_function import ICostFunction


class NeuralNetwork():
    def __init__(self,
                 neurons_count_per_layer: List[int],
                 activation_function: IActivationFunction,
                 cost_function: ICostFunction,
                 random_seed: int = None):
        if random_seed is not None:
            np.random.seed(random_seed)

        self.layers_count = len(neurons_count_per_layer)
        self.weights = [
            np.random.normal(
                size=(neurons_count_per_layer[layer_number],
                      neurons_count_per_layer[layer_number + 1]))
            for layer_number in range(0, self.layers_count - 1)
        ]
        self.biases = [
            np.random.normal(size=(neurons_count_per_layer[layer_number], 1))
            for layer_number in range(0, self.layers_count - 1)
        ]
        self.activation_function = activation_function
        self.cost_function = cost_function

    def _feedforward(self, x: np.array):
        for weight, bias in zip(self.weights, self.biases):
            x = self.activation_function.calculate_value(
                np.dot(weight, x) + bias)

        return x

    def get_cost_function_value(self, X: np.array, y: np.array):
        activation_function_values = [
            self._feedforward(observation) for observation in X
        ]
        activation_function_values = np.concatenate(activation_function_values)
        return self.cost_function.calculate_value(y,
                                                  activation_function_values)

    def _collect_activation_function_arguments_and_values(
            self, x: np.array) -> Tuple[np.array, np.array]:
        activation_function_arguments = list()
        activation_function_values = [x]

        current_activation_value = x
        for weight, bias in zip(self.weights, self.biases):
            current_activation_argument = np.dot(
                weight, current_activation_value) + bias
            current_activation_value = self.activation_function.calculate_value(
                current_activation_argument)
            activation_function_arguments.append(current_activation_argument)
            activation_function_values.append(current_activation_value)

        return activation_function_arguments, activation_function_values

    def _backpropagation(self, x: np.array, y: np.array):
        activation_function_arguments, activation_function_values = (
            self._collect_activation_function_arguments_and_values(x))

        output_error = np.multiply(
            self.cost_function.calculate_derivative_value(
                y, activation_function_values[-1]),
            self.activation_function.calculate_derivative_value(
                activation_function_arguments[-1]),
        )

        errors = np.zeros(self.biases.shape)
        errors[-1] = output_error
        for layer in range(2, self.layers_count):
            errors[-layer] = np.multiply(
                np.dot(
                    np.transpose(self.weights[-layer + 1]),
                    errors[-layer + 1]
                ),
                self.activation_function.calculate_derivative_value(
                    activation_function_arguments[-layer]
                )
            )

        biases_gradient = np.zeros(self.biases.shape)
        weights_gradient = np.zeros(self.weights.shape)
        for layer in range(1, self.layers_count):
            weights_gradient[-layer] = np.dot(
                activation_function_values[-layer - 1],
                np.transpose(errors[-layer])
            )
            biases_gradient[-layer] = errors[-layer]

        return weights_gradient, biases_gradient

    def _gradient_descent(self, samples: List[np.array],
                          labels: List[np.array], learning_rate: float):

        weight_gradient = [np.zeros(weight.shape) for weight in self.weights]
        biases_gradient = [np.zeros(bias.shape) for bias in self.biases]
        for sample, label in zip(samples, labels):
            sample_weights_gradient, sample_biases_gradient = self._backpropagation(
                sample, label)
            weight_gradient = np.add(weight_gradient, sample_weights_gradient)
            biases_gradient = np.add(biases_gradient, sample_biases_gradient)

        self.weights = np.subtract(
            self.weights, (learning_rate / len(samples)) * weight_gradient)
        self.biases = np.subtract(
            self.biases, (learning_rate / len(samples)) * biases_gradient)
