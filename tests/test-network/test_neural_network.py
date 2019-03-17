import pytest
import numpy as np

from backpropagation.network.neural_network import NeuralNetwork
from backpropagation.network.activation_function import (
    IdentityActivationFunction, SigmoidActivationFunction)
from backpropagation.network.cost_function import (MSECostFunction,
                                                   SECostFunction)
from backpropagation.network.regularizator import L1Regularizator
from numpy.testing import assert_array_equal, assert_allclose

feedforward_layers = [[1, 1], [2, 2, 1], [2, 3, 1]]

feedforward_weights = [
    [np.array([-1, 1])],
    [np.array([[-3, 1, 1], [0, 2, -1]]),
     np.array([[0, 1, -1]])],
    [np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]]),
     np.array([[1, 0, 0, 1]])]
]

feedforward_activation = [
    SigmoidActivationFunction(),
    SigmoidActivationFunction(),
    IdentityActivationFunction(),
]

feedforward_input = [np.array([1]), np.array([1, 2]), np.array([1, 1])]

feedforward_expected = [
    np.array([0.5]),
    np.array([0.5]),
    np.array([3]),
]

feedforward_test_cases = list(
    zip(feedforward_layers, feedforward_weights, feedforward_activation,
        feedforward_input, feedforward_expected))


@pytest.mark.parametrize(
    "neurons_count_per_layer, weights, activation, test_input, expected",
    feedforward_test_cases)
def test_neural_network_feedforward(neurons_count_per_layer, weights,
                                    activation, test_input, expected):
    nn = NeuralNetwork(
        neurons_count_per_layer=neurons_count_per_layer,
        activation_function=activation,
        cost_function=MSECostFunction())

    nn.weights = weights
    assert_array_equal(nn._feedforward(test_input), expected)


@pytest.mark.parametrize(
    "neurons_count_per_layer, weights, X_input, y_input, expected",
    [([1, 1], [np.array([[-1, 1]])], np.array([[1]]), np.array([[1]]), 0.125)])
def test_neural_network_cost_value(neurons_count_per_layer, weights, X_input,
                                   y_input, expected):
    nn = NeuralNetwork(
        neurons_count_per_layer=neurons_count_per_layer,
        activation_function=SigmoidActivationFunction(),
        cost_function=SECostFunction())

    nn.weights = weights

    result = nn.get_cost_function_value(X_input, y_input)
    assert_array_equal(result, expected)


backpropagation_layers = [[2, 2, 2], [2, 3, 1]]

backpropagation_weights = [[
    np.array([[0, 1, 1], [1, 2, 2]]),
    np.array([[1, 2, 2], [0, 1, 1]])
], [np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]]),
    np.array([[1, 0, 0, 1]])]]

backpropagation_X = [np.array([[2, 3]]), np.array([[1, 1]])]

backpropagation_y = [np.array([[0, 1]]), np.array([[2]])]

backpropagation_expected = [[
    np.array([[81, 162, 243], [81, 162, 243]]),
    np.array([[33, 165, 363], [15, 75, 165]])
], [np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]),
    np.array([[1, 3, 1, 2]])]]

backpropagation_test_cases = list(
    zip(backpropagation_layers, backpropagation_weights, backpropagation_X,
        backpropagation_y, backpropagation_expected))


@pytest.mark.parametrize("neurons_count_per_layer, weights, X, y, expected",
                         backpropagation_test_cases)
def test_neural_network_backpropagation(neurons_count_per_layer, weights, X, y,
                                        expected):
    nn = NeuralNetwork(
        neurons_count_per_layer=neurons_count_per_layer,
        activation_function=IdentityActivationFunction(),
        cost_function=MSECostFunction())

    nn.weights = weights

    gradient = nn._backpropagation(X.T, y.T)

    for x, y in zip(gradient, expected):
        assert_array_equal(x, y)


gradient_descent_layers = [[2, 2, 2], [2, 3, 1], [2, 3, 1], [2, 3, 1]]

gradient_descent_weights = [
    [np.array([[0, 1, 1], [1, 2, 2]]),
     np.array([[1, 2, 2], [0, 1, 1]])],
    [np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]]),
     np.array([[1, 0, 0, 1]])],
    [np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]]),
     np.array([[1, 0, 0, 1]])],
    [np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]]),
     np.array([[1, 0, 0, 1]])]
]

gradient_descent_X = [
    np.array([[2, 3]]),
    np.array([[1, 1]]),
    np.array([[1, 1], [1, 1]]),
    np.array([[1, 1], [1, 1]])
]

gradient_descent_y = [
    np.array([[0, 1]]),
    np.array([[2]]),
    np.array([[2], [2]]),
    np.array([[2], [2]])
]

gradient_descent_expected = [[
    np.array([[-81, -161, -242], [-80, -160, -241]]),
    np.array([[-32, -163, -361], [-15, -74, -164]])
], [np.array([[1, 1, 1], [0, 0, 1], [0, 0, -1]]),
    np.array([[0, -3, -1, -1]])],
                             [
                                 np.array([[1, 1, 1], [0, 0, 1], [0, 0, -1]]),
                                 np.array([[0, -3, -1, -1]])
                             ],
[
                                 np.array([[1, .5, .5], [0, 0, .5], [0, -.5, -1]]),
                                 np.array([[0, -3, -1, -1.5]])
                             ]
]

gradient_descent_regularizator = [
    None, None, None, L1Regularizator()
]

gradient_descent_test_cases = list(
    zip(gradient_descent_layers, gradient_descent_weights, gradient_descent_X,
        gradient_descent_y, gradient_descent_regularizator, gradient_descent_expected))


@pytest.mark.parametrize("neurons_count_per_layer, weights, X, y, regularizator, expected",
                         gradient_descent_test_cases)
def test_neural_network_gradient_descent(neurons_count_per_layer, weights, X,
                                         y, regularizator, expected):
    nn = NeuralNetwork(
        neurons_count_per_layer=neurons_count_per_layer,
        activation_function=IdentityActivationFunction(),
        cost_function=SECostFunction(),
        regularizator=regularizator)

    nn.weights = weights
    nn._stochastic_gradient_descent(
        samples=X,
        labels=y,
        learning_rate=1,
        regularization_param=1,
        mini_batch_size=10,
        epochs_count=1)
    for x, y in zip(nn.weights, expected):
        assert_allclose(x, y)
