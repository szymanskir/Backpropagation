import numpy as np

from itertools import accumulate
from functools import reduce
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
        calculate_activation = self.activation_function.calculate_value
        x = reduce(
            lambda z, w: calculate_activation(w @ np.insert(z, 0, 1)),
            [x] + self.weights
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
        def single_pass(previous_parameters, weight):
            _, value = previous_parameters
            argument = weight @ np.insert(value, 0, 1, axis=0)
            value = self.activation_function.calculate_value(argument)
            return argument, value

        activation_parameters = list(accumulate(
            [(None, x)] + self.weights, single_pass)
        )
        activation_arguments, activation_values = zip(*activation_parameters)
        activation_arguments = [*activation_arguments][1:]
        activation_values = [*activation_values]

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

        def propagate_backwards(errors, layer):
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

        gradient = [d[1] for d in deltas[1:]]
        gradient.reverse()
        return gradient

    def _gradient_descent(
            self,
            samples: np.array,
            labels: np.array,
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

        gradient = self._backpropagation(samples.T, labels.T)
        self.weights = [w - learning_rate * g
                        for w, g in zip(self.weights, gradient)]

    def _stochastic_gradient_descent(
            self,
            samples: np.array,
            labels: np.array,
            learning_rate: float = 0.03,
            mini_batch_size: int = 10,
            epochs_count: int = 10,
            test_samples: np.array = None,
            test_labels: np.array = None,
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

                epochs_count (int):
                    Number of epochs used during training.

                test_samples (List[np.array]):
                    List of inputs used for neural network testing.

                test_labels (List[np.array]):
                    List of expected outputs used for neural network testing.       
        """
        training_data = list(zip(samples, labels))
        mini_batch_count = len(training_data)//mini_batch_size
        train_cost = list()
        test_cost = list()
        for epoch in range(epochs_count):
            train_cost.append(self.get_cost_function_value(samples, labels))
            np.random.shuffle(training_data)
            print(f'{epoch} epoch...')
            for mini_batch in range(mini_batch_count):
                start_index = mini_batch * mini_batch_size
                training_batch = training_data[
                    start_index:start_index + mini_batch_size]
                samples, labels = map(list, zip(*training_batch))
                self._gradient_descent(samples, labels, learning_rate)

            if test_samples and test_labels:
                test_cost.append(self.get_cost_function_value(
                    test_samples, test_labels))

        return train_cost, test_cost
