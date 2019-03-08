import pytest
import numpy as np

from backpropagation.network.neural_network import NeuralNetwork
from backpropagation.network.activation_function import (
    IdentityActivationFunction, SigmoidActivationFunction)
from backpropagation.network.cost_function import (MSECostFunction,
                                                   SECostFunction)
from numpy.testing import assert_array_equal, assert_allclose


@pytest.mark.parametrize(
    "neurons_count_per_layer, weights, test_input, expected",
    [
        (
            # Test case 1
            [1, 1],
            [np.array([-1, 1])],
            np.array([1]),
            0.5),

        # Test case 2
        ([2, 2, 1],
         [np.array([[-3, 1, 1], [0, 2, -1]]),
          np.array([[0, 1, -1]])], np.array([1, 2]), 0.5)
    ])
def test_neural_network_feedforward(neurons_count_per_layer, weights,
                                    test_input, expected):
    nn = NeuralNetwork(
        neurons_count_per_layer=neurons_count_per_layer,
        activation_function=SigmoidActivationFunction(),
        cost_function=MSECostFunction())

    nn.weights = weights
    assert_array_equal(nn._feedforward(test_input), expected)


@pytest.mark.parametrize(
    "neurons_count_per_layer, weights, X_input, y_input, expected",
    [([1, 1], [np.array([[-1, 1]])], [np.array([1])], [np.array([1])], 0.125)])
def test_neural_network_cost_value(neurons_count_per_layer, weights, X_input,
                                   y_input, expected):
    nn = NeuralNetwork(
        neurons_count_per_layer=neurons_count_per_layer,
        activation_function=SigmoidActivationFunction(),
        cost_function=SECostFunction())

    nn.weights = weights

    result = nn.get_cost_function_value(X_input, y_input)
    assert_array_equal(result, expected)


def test_neural_network_backpropagation():
    nn = NeuralNetwork(
        neurons_count_per_layer=[2, 2, 2],
        activation_function=IdentityActivationFunction(),
        cost_function=MSECostFunction())

    nn.weights = [
        np.array([[0, 1, 1], [1, 2, 2]]),
        np.array([[1, 2, 2], [0, 1, 1]])
    ]

    X = np.array([2, 3])
    y = np.array([0, 1])

    gradient = nn._backpropagation(X, y)
    assert_array_equal(
        gradient,
        np.array([[[81, 162, 162], [81, 243, 243]],
                  [[33, 165, 75], [15, 363, 165]]]))


def test_neural_network_gradient_descent():
    nn = NeuralNetwork(
        neurons_count_per_layer=[2, 2, 2],
        activation_function=IdentityActivationFunction(),
        cost_function=SECostFunction())

    nn.weights = [
        np.array([[0, 1, 1], [1, 2, 2]]),
        np.array([[1, 2, 2], [0, 1, 1]])
    ]

    X = [np.array([2, 3])]
    y = [np.array([0, 1])]

    nn._gradient_descent(X, y, 0.5)
    assert_allclose(
        nn.weights,
        np.array([[[-40.5, -80, -80], [-39.5, -119.5, -119.5]],
                  [[-15.5, -80.5, -35.5], [-7.5, -180.5, -81.5]]]))


def test_neural_network_stochastic_gradient_descent():
    nn = NeuralNetwork(
        neurons_count_per_layer=[2, 5, 5, 4, 4, 1],
        activation_function=SigmoidActivationFunction(),
        cost_function=MSECostFunction())

    X = list(map(np.array, [[1, 0], [1, 1], [0, 1], [0, 0]]))
    y = list(map(np.array, [[0], [1], [0], [1]]))
    cost = nn._stochastic_gradient_descent(X, y, epocs_count=1000, learning_rate=0.1)

    X = [[0.1, 0], [0.2, 0.1], [0.3, 0], [0.11, 0.14], [0.21, 0.20],
         [0.31, 0.50], [0.30, 0.30], [0.51, 0.51], [0.18, 0.22], [0.1, 0],
         [0.123, 0.2131231], [0, 0.1], [0, 0.2], [0.123, 0.16],
         [0.12312, 0.1231231], [0.1249124, 0.123912931]]

    y = [[0.1, 0], [0.2, 0.1], [0.3, 0], [0.11, 0.14], [0.21, 0.20],
         [0.31, 0.50], [0.30, 0.30], [0.51, 0.51], [0.18, 0.22], [0.1, 0],
         [0.123, 0.2131231], [0, 0.1], [0, 0.2], [0.123, 0.16],
         [0.12312, 0.1231231], [0.1249124, 0.123912931]]

    cost = nn._stochastic_gradient_descent(
        X, y, epocs_count=200, mini_batch_size=16)
