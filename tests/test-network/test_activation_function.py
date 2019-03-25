import pytest
import numpy as np

from backpropagation.network.activation_function import (
    IdentityActivationFunction,
    ReLUActivationFunction,
    SigmoidActivationFunction)
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("test_input, expected", [
    (np.array([0]), np.array([0.5])),
    (np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])),
    (np.array([0, np.log(1 / 3), np.log(1 / 7)]), np.array([0.5, 0.25, 0.125
                                                            ])),
])
def test_sigmoid_function_value(test_input, expected):
    sigmoid_activation_function = SigmoidActivationFunction()
    result = sigmoid_activation_function.calculate_value(test_input)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("test_input, expected", [
    (np.array([0]), np.array([0.25])),
    (np.array([0, 0, 0]), np.array([0.25, 0.25, 0.25])),
    (np.array([0, np.log(1 / 3), np.log(1 / 7)]),
     np.array([0.25, 0.1875, 0.109375])),
])
def test_sigmoid_function_derivative_value(test_input, expected):
    sigmoid_activation_function = SigmoidActivationFunction()
    result = sigmoid_activation_function.calculate_derivative_value(test_input)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("test_input, expected", [
    (np.array([[113]]), np.array([[113]])),
    (np.array([1, 13, -1]), np.array([1, 13, -1])),
    (np.array([[1, 13], [-1, 3]]), np.array([[1, 13], [-1, 3]]))
])
def test_identity_function_value(test_input, expected):
    identity_activation_function = IdentityActivationFunction()
    result = identity_activation_function.calculate_value(test_input)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("test_input, expected", [
    (np.array([[113]]), np.array([[1]])),
    (np.array([1, 13, -1]), np.array([1, 1, 1])),
    (np.array([[1, 13], [-1, 3]]), np.array([[1, 1], [1, 1]]))
])
def test_identity_function_derivative_value(test_input, expected):
    identity_activation_function = IdentityActivationFunction()
    result = identity_activation_function.calculate_derivative_value(
        test_input)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("test_input, expected", [
    (np.array([[-1]]), np.array([[0]])),
    (np.array([[1, -1, 3]]), np.array([[1, 0, 3]])),
    (np.array([[-123, 0], [123, 3]]), np.array([[0, 0], [123, 3]])),
])
def test_relu_function_value(test_input, expected):
    relu_activation_function = ReLUActivationFunction()
    result = relu_activation_function.calculate_value(test_input)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("test_input, expected", [
    (np.array([[-1]]), np.array([[0]])),
    (np.array([[1, -1, 3]]), np.array([[1, 0, 1]])),
    (np.array([[-123, 0], [123, 3]]), np.array([[0, 1], [1, 1]])),
])
def test_relu_function_derivative_value(test_input, expected):
    relu_activation_function = ReLUActivationFunction()
    result = relu_activation_function.calculate_derivative_value(
        test_input)

    assert_array_equal(result, expected)
