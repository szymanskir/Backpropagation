import pytest
import numpy as np

from backpropagation.network.neural_network import NeuralNetwork
from backpropagation.network.activation_function import (
    IdentityActivationFunction,
    SigmoidActivationFunction)
from backpropagation.network.cost_function import (MSECostFunction,
                                                   SECostFunction)
from numpy.testing import assert_array_equal


@pytest.mark.parametrize(
    "neurons_count_per_layer, weights, biases, test_input, expected",
    [([1, 1], [np.array([[1]])], [np.array([[-1]])], np.array([[1]]),
      np.array([[0.5]])),
     ([2, 2, 1], [np.array([[1, 1], [2, -1]]),
                  np.array([[1, -1]])],
      [np.array([[-3], [0]]), np.array([[0]])], np.array([[1], [2]]),
      np.array([[0.5]]))])
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
    "neurons_count_per_layer, weights, biases, X_input, y_input, expected",
    [([1, 1], [np.array([[1]])], [np.array([[-1]])], np.array([[1]]),
      np.array([[1]]), 0.125)])
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
        cost_function=SECostFunction())

    nn.weights = np.array([
        [[1, 1], [2, 2]],
        [[2, 2], [1, 1]]
    ])
    nn.biases = np.array([[[0], [1]], [[1], [0]]])

    X = np.array([[2], [3]])
    y = np.array([[0], [1]])

    weights_gradient, biases_gradient = nn._backpropagation(X, y)
    assert_array_equal(weights_gradient, np.array([
        [[162, 162], [243, 243]],
        [[165, 75], [363, 165]]
    ]))
    assert_array_equal(biases_gradient, np.array([
        [[81], [81]], [[33], [15]]
    ]))
