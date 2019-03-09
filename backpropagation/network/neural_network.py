import numpy as np

from typing import List, Tuple
from .activation_function import IActivationFunction
from .cost_function import ICostFunction


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
            random_seed: int = None
    ):

        if random_seed is not None:
            np.random.seed(random_seed)

        self.layers_count = len(neurons_count_per_layer)

        self.weights = [
            np.random.normal(size=(
                neurons_count_per_layer[layer + 1],
                neurons_count_per_layer[layer] + 1
            ))
            for layer in range(self.layers_count - 1)
        ]

        self.activation_function = activation_function
        self.cost_function = cost_function

    def _feedforward(self, x: np.array) -> np.array:
        """Calculates the output of the Neural network for the given input.

            Args:
                x (np.array): Value of the input.

            Returns (np. array):
                The output values of the neural network for the given input.

        """
        for weight in self.weights:
            x = self.activation_function.calculate_value(
                weight @ np.insert(x, 0, 1)
            )

        return x

    def get_cost_function_value(self, X: List[np.array], y: List[np.array]):
        """Calculates the cost value of the neural network for the given inputs.

            Args:
                X (List[np.array]): List of inputs.
                y (List[np.array]): List of expected outputs.

            Returns (int):
                The cost value of the neural network for the given inputs.
        """
        activation_values = [
            self._feedforward(observation) for observation in X
        ]
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
        activation_arguments = list()
        activation_values = [x]

        value = x
        for weight in self.weights:
            argument = weight @ np.insert(value, 0, 1)
            value = self.activation_function.calculate_value(argument)
            activation_arguments.append(argument)
            activation_values.append(value)

        return activation_arguments, activation_values

    def _backpropagation(
            self,
            x: np.array,
            y: np.array
    ) -> Tuple[np.array, np.array]:
        """Performs backpropagation and calculates gradient for weights and biases.

            Args:
                x (np.array): Neural network input.
                y (np. array): Expected neural network output.

            Returns (Tuple[np.array, np.array]):
                Tuple compoes of the gradient for weights and gradient
                for biases.
        """
        activation_function_arguments, activation_function_values = (
            self._collect_activation_function_arguments_and_values(x))

        output_error = np.multiply(
            self.cost_function.calculate_derivative_value(
                y, activation_function_values[-1]),
            self.activation_function.calculate_derivative_value(
                activation_function_arguments[-1]),
        )

        gradient = list()
        error = output_error
        weight_gradient = (
            np.atleast_2d(activation_function_values[-2]).transpose() @
            np.atleast_2d(error)
        )

        gradient.append(np.insert(weight_gradient.transpose(), 0, error, axis=1))
        for layer in range(2, self.layers_count):
            activation_derivative = self.activation_function.calculate_derivative_value(
                activation_function_arguments[-layer]
            )

            weight = self.weights[-layer + 1][:, 1:]
            error = weight.transpose() @ error * activation_derivative

            weight_gradient = (
                np.atleast_2d(activation_function_values[-layer - 1]).transpose() @
                np.atleast_2d(error)
            )
            gradient.insert(0, np.insert(weight_gradient.transpose(), 0, error, axis=1))

        return gradient

    def _gradient_descent(
            self,
            samples: List[np.array],
            labels: List[np.array],
            learning_rate: float
    ):
        """Performs gradient descent and updates network weights and biases.

           The gradient is calculated using the backpropagation method.

        Args:
            samples (List[np.array]):
                List of inputs used for minimizing the cost function.

            labels (List[np.array]):
                List of expected outputs used for minimizing the cost function.
        """

        gradient = [np.zeros(weight.shape) for weight in self.weights]
        for sample, label in zip(samples, labels):
            gradient_sample = self._backpropagation(sample, label)
            gradient = [np.add(x, y)
                        for x, y in zip(gradient, gradient_sample)]

        self.weights = [np.subtract(x, ((learning_rate)/len(samples)) * y)
                        for x, y in zip(self.weights, gradient)]

    def _stochastic_gradient_descent(
            self,
            samples: np.array,
            labels: np.array,
            learning_rate: float = 0.5,
            mini_batch_size: int = 10,
            epocs_count: int = 10
    ):
        """Optimizes the neural network weights and biases using SGD.

            Args:
                samples (List[np.array]):
                    List of inputs used for neural network training.

                labels (List[np.array]):
                    List of expected outputs used for neural network training.

                learning_rate (float):
                    Learning rate coefficient.

                mini_batch_size (int):
                    Size of mini batches used during gradient descent.

                epocs_count (int):
                    Number of epocs used during training.
        """
        training_data = list(zip(samples, labels))
        mini_batch_count = len(training_data)//mini_batch_size
        cost = list()
        for epoc in range(epocs_count):
            cost.append(self.get_cost_function_value(samples, labels))
            np.random.shuffle(training_data)
            print(f'{epoc} epoc...')
            for mini_batch in range(mini_batch_count):
                start_index = mini_batch*mini_batch_size
                training_batch = training_data[
                    start_index:start_index + mini_batch_size]
                samples, labels = map(list, zip(*training_batch))
                self._gradient_descent(samples, labels, learning_rate)

        return cost
