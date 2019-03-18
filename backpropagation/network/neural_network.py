import logging
import numpy as np

from itertools import accumulate
from functools import reduce
from math import ceil
from typing import List, Tuple
from .activation_function import IActivationFunction
from .cost_function import ICostFunction
from .regularizator import IRegulatizator

logger = logging.getLogger(__name__)


class NeuralNetwork():
    """ Feedforward neural network using SGD for training.

        Attributes:
            layers_count (int): Number of layers in the network.

            activation_function (IActivationFunction):
                Function used for calculating the neurons
                activation in each layer.

            cost_function (ICostFunction):
                Function used for calculating the cost of the neural network.

            weights (np.array):
                Neural network weight and biases values.
    """

    def __init__(
            self,
            neurons_count_per_layer: List[int],
            activation_function: IActivationFunction,
            cost_function: ICostFunction,
            regularizator: IRegulatizator = None,
            random_seed: int = None
    ):

        if random_seed is not None:
            np.random.seed(random_seed)

        self.layers_count = len(neurons_count_per_layer)

        self.weights = [
            self._create_weight_matrix(size=(
                neurons_count_per_layer[layer + 1],
                neurons_count_per_layer[layer]
            ))
            for layer in range(self.layers_count - 1)
        ]
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.regularizator = regularizator

    def _create_weight_matrix(self, size: Tuple[int, int]) -> np.array:
        """Creates a weight matrix between layers of sizes specified by ``size``.

            It creates the biases weights and stacks it horizontally in front of the
            weight matrix.

            Args:
                size (Tuple[int, int]): Size of the layers between which the matrix
                should be initialized.

            Returns (np.array): Weight matrix used to calculate the activation from one
                layer to another.
        """
        rows, columns = size
        biases = np.random.normal(size=(rows, 1))
        weight_matrix = np.random.normal(size=(rows, columns), scale=1/np.sqrt(columns))

        return np.hstack([biases, weight_matrix])

    def _feedforward(self, X: np.array) -> np.array:
        """Calculates the output of the Neural network for the given input.

            Args:
                x (np.array): Value of the input.

            Returns (np. array):
                The output values of the neural network for the given input.

        """
        calculate_activation = self.activation_function.calculate_value
        output = reduce(
            lambda x, w: calculate_activation(w @ np.insert(x, 0, 1, axis=0)),
            [X] + self.weights
        )

        return output

    def get_cost_function_value(self, x: np.array, y: np.array):
        """Calculates the cost value of the neural network for the given inputs.

            Args:
                X (List[np.array]): List of inputs.
                y (List[np.array]): List of expected outputs.

            Returns (int):
                The cost value of the neural network for the given inputs.
        """
        activation_values = self._feedforward(x)
        return self.cost_function.calculate_value(y, activation_values)

    def _collect_activation_function_arguments_and_values(
            self,
            x: np.array
    ) -> Tuple[List[np.array], List[np.array]]:
        """Collects the activation function arguments and values for each layer.

            Args:
                x (np.array):
                    Input of the neural network.

            Returns (Tuple[np.array, np.array]):
                Activation function arguments and values for the given
                input for each layer.
        """
        def single_pass(previous_parameters, weight):
            _, value = previous_parameters
            argument = weight @ np.insert(value, 0, 1, axis=0)
            value = self.activation_function.calculate_value(argument)
            return argument, value

        activation_parameters = list(accumulate(
            [(None, x)] + self.weights, single_pass)
        )
        activation_arguments, activation_values = zip(*activation_parameters)
        activation_arguments = list(activation_arguments)[1:]
        activation_values = list(activation_values)

        return activation_arguments, activation_values

    def _backpropagation(
            self,
            x: np.array,
            y: np.array
    ) -> List[np.array]:
        """Performs backpropagation and calculates gradient for weights and biases.

            Args:
                x (np.array): Neural network input.
                y (np.array): Expected neural network output.

            Returns (Tuple[np.array, np.array]):
                Tuple compoes of the gradient for weights and gradient
                for biases.
        """
        activation_function_arguments, activation_function_values = (
            self._collect_activation_function_arguments_and_values(x))

        def propagate_backwards(
                errors: Tuple[np.array, np.array],
                layer: int
        ) -> Tuple[np.array, np.array]:
            error, _ = errors
            activation_derivative = self.activation_function.calculate_derivative_value(
                activation_function_arguments[-layer])

            if layer == 1:
                cost_derivative = self.cost_function.calculate_derivative_value(
                    y, activation_function_values[-layer])
                error = activation_derivative * cost_derivative
            else:
                weight = self.weights[-layer + 1][:, 1:]
                error = weight.transpose() @ error * activation_derivative

            weight_gradient = (
                error @
                activation_function_values[-layer - 1].T
            )

            bias_gradient = np.sum(error, axis=1, keepdims=True)
            gradient = np.hstack((bias_gradient, weight_gradient)) / x.shape[1]

            return error, gradient

        deltas = list(accumulate(
            [(None, None)] + list(range(1, self.layers_count)),
            propagate_backwards
        ))

        gradient = [d[1] for d in reversed(deltas[1:])]
        return gradient

    def _update_weights(
            self,
            gradient: List[np.array],
            learning_rate: float,
            regularization_param: float,
            samples_count: int):

        if self.regularizator is not None:
            only_weights = [w[:, 1:] for w in self.weights]
            regularization_terms = self.regularizator.get_derivative_terms(
                regularization_param, only_weights, samples_count)
            self.weights = [w - learning_rate * (g + np.insert(r, 0, 0, axis=1))
                            for w, g, r in zip(self.weights,
                                               gradient,
                                               regularization_terms)]
        else:
            self.weights = [w - learning_rate * g
                            for w, g in zip(self.weights, gradient)]

    def _stochastic_gradient_descent(
            self,
            samples: np.array,
            labels: np.array,
            learning_rate: float = 0.1,
            regularization_param: float = 5.0,
            mini_batch_size: int = 10,
            epochs_count: int = 10,
            test_samples: np.array = None,
            test_labels: np.array = None,
    ):
        """Optimizes the neural network weights and biases using SGD.

            Args:
                samples (np.array):
                    Inputs used for neural network training.

                labels (np.array):
                    Expected outputs used for neural network training.

                learning_rate (float):
                    Learning rate coefficient.

                regularization_param (float):
                    Regularization parameter.

                mini_batch_size (int):
                    Size of mini batches used during gradient descent.

                epochs_count (int):
                    Number of epochs used during training.

                test_samples ([np.array]):
                    Inputs used for neural network testing.

                test_labels ([np.array]):
                    Expected outputs used for neural network testing.
        """
        mini_batch_count = ceil(len(samples)/mini_batch_size)
        train_cost = list()
        test_cost = list()
        for epoch in range(epochs_count):
            train_cost.append(self.get_cost_function_value(samples.T, labels.T))

            if test_samples is not None and test_labels is not None:
                cost = self.get_cost_function_value(test_samples.T, test_labels.T)
                logger.info(f'Test cost: {cost}...')
                test_cost.append(cost)

            index_permutation = np.random.permutation(len(samples))
            samples = samples[index_permutation]
            labels = labels[index_permutation]

            logger.info(f'Calculating {epoch} epoch...')
            for sample_batch, label_batch in zip(
                    np.array_split(samples, mini_batch_count),
                    np.array_split(labels, mini_batch_count),
            ):
                gradient = self._backpropagation(sample_batch.T,
                                                 label_batch.T)
                self._update_weights(gradient,
                                     learning_rate,
                                     regularization_param,
                                     samples.shape[1])

        return train_cost, test_cost
