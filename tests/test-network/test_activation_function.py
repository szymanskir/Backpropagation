import pytest
import numpy as np

from backpropagation.network.activation_function import (
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
