import pytest
import numpy as np

from backpropagation.network.neural_network import NeuralNetwork
from backpropagation.network.activation_function import (
    IdentityActivationFunction, SigmoidActivationFunction)
from backpropagation.network.cost_function import (MSECostFunction,
                                                   SECostFunction)
from numpy.testing import assert_array_equal, assert_allclose


@pytest.mark.parametrize(
    "neurons_count_per_layer, weights, biases, test_input, expected",
    [([1, 1], np.array([1]), np.array([-1]), np.array([1]), np.array([0.5])),
     ([2, 2, 1], np.array([[[1, 1], [2, -1]], [[1, -1]]]),
      np.array([[-3, 0], [0]]), np.array([1, 2]), np.array([0.5]))])
def test_neural_network_feedforward(neurons_count_per_layer, weights, biases,
                                    test_input, expected):
    nn = NeuralNetwork(
        neurons_count_per_layer=neurons_count_per_layer,
        activation_function=SigmoidActivationFunction(),
        cost_function=MSECostFunction())

    nn.weights = weights
    nn.biases = biases

    assert_array_equal(nn._feedforward(test_input), expected)


@pytest.mark.parametrize(
    "neurons_count_per_layer, weights, biases, X_input, y_input, expected", [(
        [1, 1],
        np.array([[1], [-1]]),
        np.array([[-1]]),
        np.array([1]),
        np.array([1]),
        0.125
    )])
def test_neural_network_cost_value(neurons_count_per_layer, weights, biases,
                                   X_input, y_input, expected):
    nn = NeuralNetwork(
        neurons_count_per_layer=neurons_count_per_layer,
        activation_function=SigmoidActivationFunction(),
        cost_function=SECostFunction())

    nn.weights = weights
    nn.biases = biases

    result = nn.get_cost_function_value(X_input, y_input)
    assert_array_equal(result, expected)


def test_neural_network_backpropagation():
    nn = NeuralNetwork(
        neurons_count_per_layer=[2, 2, 2],
        activation_function=IdentityActivationFunction(),
        cost_function=MSECostFunction())

    nn.weights = np.array([[[1, 1], [2, 2]], [[2, 2], [1, 1]]])
    nn.biases = np.array([[0, 1], [1, 0]])

    X = np.array([2, 3])
    y = np.array([0, 1])

    weights_gradient, biases_gradient = nn._backpropagation(X, y)
    assert_array_equal(
        weights_gradient,
        np.array([[[162, 162], [243, 243]], [[165, 75], [363, 165]]]))
    assert_array_equal(biases_gradient, np.array([[81, 81], [33, 15]]))


def test_neural_network_gradient_descent():
    nn = NeuralNetwork(
        neurons_count_per_layer=[2, 2, 2],
        activation_function=IdentityActivationFunction(),
        cost_function=SECostFunction())

    nn.weights = np.array([[[1, 1], [2, 2]], [[2, 2], [1, 1]]])
    nn.biases = np.array([[0, 1], [1, 0]])

    X = [np.array([2, 3])]
    y = [np.array([0, 1])]

    nn._gradient_descent(X, y, 0.5)
    assert_allclose(
        nn.weights,
        np.array([[[-80, -80], [-119.5, -119.5]],
                  [[-80.5, -35.5], [-180.5, -81.5]]]))


def test_neural_network_stochastic_gradient_descent():
    nn = NeuralNetwork(
        neurons_count_per_layer=[2, 2, 2],
        activation_function=SigmoidActivationFunction(),
        cost_function=MSECostFunction())

    nn.weights = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])

    X = np.array([[[0.1], [0]], [[0.2], [0.1]], [[0.3], [0]], [[0.11], [0.14]],
                  [[0.21], [0.20]], [[0.31], [0.50]], [[0.30], [0.30]],
                  [[0.51], [0.51]], [[0.18], [0.22]], [[0.1], [0]],
                  [[0.123], [0.2131231]], [[0], [0.1]], [[0], [0.2]],
                  [[0.123], [0.16]], [[0.12312], [0.1231231]],
                  [[0.1249124], [0.123912931]]])

    y = np.array([[[0.1], [0]], [[0.2], [0.1]], [[0.3], [0]], [[0.11], [0.14]],
                  [[0.21], [0.20]], [[0.31], [0.50]], [[0.30], [0.30]],
                  [[0.51], [0.51]], [[0.18], [0.22]], [[0.1], [0]],
                  [[0.123], [0.2131231]], [[0], [0.1]], [[0], [0.2]],
                  [[0.123], [0.16]], [[0.12312], [0.1231231]],
                  [[0.1249124], [0.123912931]]])

    nn._stochastic_gradient_descent(X, y)
